# M17 — RISC-V Vector (RVV 1.0) backend with vector-length-agnostic kernels

> Plan: [PLAN.md › Phase 7](../PLAN.md#phase-7--embedded-backends-large-incremental) · Depends on: [M15](M15-emulator-targets.md) (and the backend pattern of [M13](M13-simd-backend.md) and [M16](M16-esp32-p4-pie-backend.md)) · Next: see [BACKLOG.md](BACKLOG.md)

## Goal

Create `sdrk-backend-rvv` for 64-bit RISC-V with the Vector extension 1.0 (`V`): the kernels `convert`, `convert_out`, `rotate`, `fir`, `fir_decimate`, `fir_interpolate`, `dot`, `mag_sq` and `fft` for `C32` and `CQ15`, written **vector-length agnostic** (one binary runs unchanged on VLEN 128, 256, 512, 1024). Verify the backend with the full suite in QEMU at several vector lengths, prove that results are **bit-identical for every VLEN**, and report instruction counts against the scalar fallback.

## Why

RVV is the third SIMD family after fixed-width SIMD (M13) and the ESP PIE unit (M16), and the only one whose register width is a property of the chip, not of the instruction set. Doing it in the plan's emulator-first way shows that the architecture copes with it: the facade, the numeric contracts, the determinism rule and the instruction-count method (no established library exists, so the comparison is the own fallback) stay unchanged.

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

**In:** backend crate `sdrk-backend-rvv` (riscv64 only), runtime and compile-time availability handling, facade feature `rvv`, VLEN matrix in QEMU user mode, system-mode target `targets/qemu-riscv64-rvv`, `xtask rvv-matrix`, `xtask check rvv`, instruction counting at several VLEN, CI jobs, devcontainer additions, `docs/backends.md` section.

**Out:** 32-bit RISC-V with `V`, the embedded subsets `Zve32f`/`Zve64f` (they lack 64-bit lanes or double precision; backlog), `quad_demod` and `fm_mod` (need a vectorised `atan2`/phase pipeline; backlog), half-precision and bit-manipulation vector extensions (`Zvfh`, `Zvbb`), the pre-ratification RVV 0.7.1 of older boards, real hardware, `benches` comparisons with libraries (none exists for RVV).

## Specification

### 0. Smoke test first

Before writing any kernel, prove the toolchain on the host with a 10-line example: a `global_asm!` function that sets `vsetvli`, adds two `f32` arrays and returns, built for `riscv64gc-unknown-linux-gnu` with the **default** target features (no `+v`) and run under `qemu-riscv64 -cpu rv64,v=true,vlen=128`. If this does not work, stop and resolve it explicitly (decisions log) before continuing; the rest of this milestone assumes that it does.

### 1. Crate, toolchain and unsafe policy

- `crates/backends/sdrk-backend-rvv`, `#![no_std]`, depends on `sdrk-core`; its code is compiled only for `target_arch = "riscv64"` (on every other target the crate is empty so that `cargo build --workspace` on x86_64, arm64 and Windows hosts still works).
- **No stable vector intrinsics exist** (`core::arch::riscv` has no vector types on stable; verify for the pinned toolchain and log the finding; if they are stable at implementation time they may be used instead, with the same VLEN-agnostic rules). Therefore every kernel's loop is a **whole function written in `global_asm!`** with the C ABI of the RISC-V psABI, called from a safe Rust wrapper that validates lengths and arguments first. Each function body starts with `.option push` / `.option arch, +v` and ends with `.option pop`, so the assembler accepts vector instructions although the crate is compiled **without** the `v` target feature (the compiler therefore never touches vector registers itself and no clobber lists are needed). Do not build this crate with `-C target-feature=+v`; nightly is not needed.
- Functions follow the psABI: vector registers, `vl`, `vtype` and `vstart` are caller-saved. `vxrm` (fixed-point rounding mode) and `frm` (floating-point rounding mode) are changed by some kernels: a function that changes one saves the old value (`csrrwi`/`csrr`) on entry and restores it before returning; no function assumes a value of either CSR on entry (verify the psABI wording and cite it in `docs/backends.md`).
- Local labels of the inner loops are named `.Lsdrk_rvv_<kernel>_inner` (used by `xtask count`).
- `unsafe` policy as in M13: `#[allow(unsafe_code)]` only in the module that declares the asm functions and calls them, a `// SAFETY:` comment on every `unsafe` block, `#![deny(clippy::undocumented_unsafe_blocks)]` in the crate root. The wrapper guarantees: every slice length is checked, pointers come from the slices, no function writes beyond `len` elements, output and input slices do not alias (guaranteed by `&mut`/`&`).
- No function uses the stack beyond the psABI minimum (R3.4).

### 2. Vector-length-agnostic programming rules (normative)

1. Every loop obtains its element count from `vsetvli` (`rd = vsetvli(remaining, e16|e32|e64, m<LMUL>, ta, ma)`) in each iteration and advances by `vl`; the last, shorter iteration is the tail. There is no scalar tail loop and no assumption about VLEN except VLEN ≥ 128 (guaranteed by `V`).
2. `vlenb` and `vsetvli x0, x0` style VLEN probing are forbidden (`xtask check rvv`, §7): code that reads the vector length can silently depend on it.
3. LMUL is chosen per kernel and documented in `docs/backends.md`; widening instructions double the register group (a `e16` → `e32` widening at `m4` uses `m8`), narrowing halves it. Register groups must be aligned as the instruction requires.
4. **Results are bit-identical for every VLEN** (this goes beyond R6, which only requires block-size invariance). It holds automatically for every kernel whose lanes are independent outputs (`convert`, `convert_out`, `rotate`, `fir*` vectorised over outputs, `mag_sq`, `fft` butterflies) because each output sees the same operations in the same order whatever the lane count. It must be **made** true for reductions: see `dot` in §3.
5. Elements are loaded with their natural alignment (`e16` at 2 bytes, `e32` at 4 bytes); never load `CQ15` pairs as `e32` (the slice may only be 2-byte aligned). Interleaved complex data is read and written with segment instructions (`vlseg2e16.v`, `vlseg2e32.v`, `vsseg2e16.v`, …) so that real and imaginary parts end up in separate register groups.
6. Fused multiply-add (`vfmacc`/`vfmadd`) is allowed in tolerance kernels only, never in bit-exact kernels (as in M13).

### 3. Kernel contracts

| Kernel | Types | Contract | RVV notes |
| --- | --- | --- | --- |
| `convert` | CU8, CS8, CS16 (and the other unpacked formats of M05) → C32, CQ15 | **bit-exact** vs fallback | widening loads (`vzext.vf2`, `vsext.vf2`, `vwadd`), shifts; packed formats may stay scalar (delegate to the fallback) |
| `convert_out` | C32, CQ15 → CU8, CS8, CS16 | **bit-exact** | the result is `floor(v + 0.5)` clamped, with the NaN/∞ rules of M08. `vfcvt.x.f.v` rounds with `frm` (use round-down after adding 0.5, or `vfcvt.rtz` plus a correction; prove equality in a test) and returns the maximum for NaN, which differs from the fallback: mask NaN lanes (`vmfne.vv v0, v, v`) and set them to 0 as the fallback does. f32 → Q15 stays in the exact semantics of R7 (the fallback computes in f64: use only formulations proven identical in the extreme-value tests, otherwise run those lanes through a slower exact path) |
| `rotate` | C32, CQ15 | as the fallback contract (C32 ≥ 120 dB, CQ15 ≥ 80 dB) | the `u32` phase accumulator stays integer: lane phases are `phase0 + vid·step` (wrapping `vmul`/`vadd` at `e32`), the next iteration starts at `phase0 + vl·step` (wrapping scalar multiply); table lookup + interpolation with `vluxei32.v` gathers |
| `fir`, `fir_decimate`, `fir_interpolate` | C32 | tolerance, ≥ 110 dB | vectorised over outputs: for each tap, `acc += h[k] · x[n−k …]` with a unit-stride (or, for decimation, strided `vlse`) load and `vfmacc.vf`; taps are summed in the same order for every lane, so the result does not depend on VLEN |
| | CQ15 | **bit-exact** | segment-load re/im, widening multiply-accumulate (`vwmacc.vx` into `e32`, then `vwadd` into `e64` accumulators where the headroom analysis of M13 §2 requires it: i32 accumulation only if the state constructor proved `Σ\|h[k]\|·32768 < 2^31`), final rounding and saturation with **`vnclip.wi … , 15` under `vxrm = rnu` (round-to-nearest-up)**, which equals R7's `(x + (1 << 14)) >> 15` with saturation; from `e64` narrow in two steps (`e64`→`e32` by 15, `e32`→`e16` by 0), which saturates exactly as the single-step fallback. Prove with the stress cases that no intermediate wraps |
| `dot` | C32 | tolerance, ≥ 110 dB (as the fallback) | **fixed 16-lane model:** the result is defined as `p[j] = Σ x[i]·y[i]` over `i ≡ j (mod 16)` (ascending `i`, FMA allowed), combined by the fixed tree `((p0+p8)+(p4+p12)) + ((p2+p10)+(p6+p14))` plus the odd lanes likewise (write the exact tree in `docs/backends.md`). Implementation: `vsetvli x0, 16, e32, m4` keeps exactly 16 lanes at every VLEN ≥ 128, the loop steps by 16, the (shorter) last step uses the leftover elements with `vl` < 16 and leaves the other lanes unchanged (`tu` policy). This makes `dot` independent of VLEN at the price that it does not get faster with a longer vector — `dot` is the only kernel with this property, and `docs/backends.md` says so |
| | CQ15 | **bit-exact** | integer accumulation is associative, so the VLEN-dependent order does not matter: widening multiply-accumulate into `e64`, one rounding at the end as in M13 |
| `mag_sq` | C32, CQ15 | C32 ≥ 140 dB, CQ15 bit-exact | segment loads, `vfmul`/`vfmacc`, `vwmul` |
| `fft` | C32, CQ15 (fixed and block floating point) | M07 / M13 contracts (C32 ≥ 110 dB; CQ15 per the M07 table; BFP exponents may differ from the fallback, compare after applying `2^exponent`) | radix-2/4 butterflies over split re/im register groups; stages with distance `d` below the vector length use strided/segment loads or `vrgather`/`vslide` permutations written so that the result is independent of VLEN (each butterfly is the same arithmetic wherever it is computed); twiddles come from the static tables of M07 via `vluxei`/`vlse`; BFP block maximum with `vredmax` over `\|x\|` (an integer reduction, order-independent) |

Kernels that need a feature the running hardware lacks, an unsupported format, or a length below a documented minimum call the fallback (never panic, R5). Slices shorter than one vector are handled by the same loop (`vl` smaller).

### 4. Availability and dispatch

- `Rvv::available() -> bool`. With feature `std` (enabled by the facade's `std` feature) it reads the `V` bit of the Linux `AT_HWCAP` auxiliary vector (through `std::arch::is_riscv_feature_detected!("v")` if that macro is stable on the pinned toolchain; otherwise through the `libc` crate behind feature `std`, logged in the decisions log and passing `cargo deny`) once per call (cached by `std`, not by us: R6) and requires `vlenb ≥ 16` (guaranteed). Without `std` the answer is a compile-time constant: `true` only if the build sets `--cfg sdrk_rvv_always` (a `RUSTFLAGS` cfg like `sdrk_fallback_only` in M13, **not** a cargo feature, because `cargo test --all-features` would switch it on for every riscv64 host without `V`); declare it in the workspace lints (`unexpected_cfgs`).
- Every kernel entry of `Rvv` calls the fallback when `available()` is false, so a riscv64 binary without `V` hardware runs correctly (tested, §6).
- Facade: feature `rvv` (default **off**, because riscv64 hosts are not yet common; the riscv64 CI jobs enable it) adds `delegate!(… => Rvv)` for the kernels above under `#[cfg(all(feature = "rvv", target_arch = "riscv64"))]`; `backend_for` returns `"rvv"` if `Rvv::available()`, otherwise `"fallback"`. `sdrk_fallback_only` (M13) still forces the fallback everywhere. `simd` and `rvv` never compete: they are compiled for disjoint architectures.

### 5. Tooling and targets

- Devcontainer and CI install: Rust targets `riscv64gc-unknown-linux-gnu` (user mode) and `riscv64gc-unknown-none-elf` (system mode), the package `gcc-riscv64-linux-gnu` (linker) and `qemu-user`/`qemu-system-misc` from Ubuntu 24.04 (QEMU 8.2, which supports RVV 1.0 with `v=true,vlen=<bits>`; record the exact version used in `docs/backends.md`). Add the missing pieces to `.devcontainer/Dockerfile`.
- `cargo xtask rvv-matrix`: builds the workspace tests for `riscv64gc-unknown-linux-gnu` with `--features rvv` and runs them (`cargo test --workspace --features rvv --target riscv64gc-unknown-linux-gnu`, runner `qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu rv64,v=true,vlen=<V>,elen=64,vext_spec=v1.0`) for **V = 128, 256, 512 and 1024**, and once with `-cpu rv64,v=false` (the fallback path; `backend_for` must report `"fallback"`). It sets `SDRK_DIGEST_OUT=target/rvv/digest-<V>.txt` for each run (see below), then compares the digest files of the four vector lengths line by line and fails if any line differs. Missing `qemu-riscv64` is an error with an install hint (`--skip-missing` turns it into a skip, for Windows hosts).
- **Digest test** (`crates/backends/sdrk-backend-rvv/tests/vlen_invariance.rs`, uses the testkit as an integration test per R8.7): runs every accelerated kernel through the facade on PCG32 inputs (seeds 500–519 as listed in the table of the milestones README) with lengths from the list in §6, computes a 64-bit FNV-1a hash over the **output bits** of each case, and — if `SDRK_DIGEST_OUT` is set — appends `<case> <hex>` lines to that file (sorted by case name). The test also compares every case with the fallback under the kernel's contract (bit-exact or SNR against the reference), so it is meaningful on its own.
- System-mode target `targets/qemu-riscv64-rvv` (tier 2, built as in M15 §1): Rust target `riscv64gc-unknown-none-elf`, `RUSTFLAGS` including `--cfg sdrk_rvv_always`, machine `qemu-system-riscv64 -machine virt -cpu rv64,v=true,vlen=<V>,elen=64 -nographic -bios none -semihosting-config enable=on,target=native -kernel <elf>`, output and exit by semihosting as the other RISC-V target. The start-up code must enable the vector unit before the suite runs: set `mstatus.VS` to Initial (`csrs mstatus, 0x200`) — without it the first vector instruction traps; document the sequence in `docs/embedded.md`. `xtask emu` gets the option `--vlen <bits>` (only valid for this target, default 256).

### 6. Verification

- Lengths for every accelerated kernel (elements): 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 4095, 4096, 4097 — they straddle `VLMAX` for every VLEN/LMUL combination, so every kernel runs its full-vector, partial-vector and single-iteration paths at each VLEN. Additional: taps 1…300 (CQ15 chunking and accumulator headroom), unaligned input offsets of one element, −32768 everywhere, full-scale f32, NaN/±∞ where defined (`convert_out`, `rotate` phase wrap).
- The existing `backend_vs_fallback_*` testkit cases run through the facade and therefore cover the backend automatically; with `sdrk_fallback_only` the whole suite still passes.
- Block-size invariance (R6.2) for the stateful kernels (`fir*`, `rotate`) with the random chunk plans of the testkit, at VLEN 128 and 512.
- The same quick suite runs bare-metal in `targets/qemu-riscv64-rvv` at VLEN 128, 256 and 512.

### 7. `xtask check rvv`

Part of `cargo xtask check`. It scans the source text of `crates/backends/sdrk-backend-rvv` (comments and Rust string contents excluded except inside `global_asm!`/`asm!` literals) and reports:
1. any use of `vlenb` (rule 2),
2. any vector mnemonic (`v…` instruction) in a `global_asm!` function body before the first `vsetvli`/`vsetivli` of that function,
3. a `global_asm!` function without `.option push` … `.option pop` around its body.
The check does nothing if the crate does not exist yet. Negative tests as in M00 (`xtask/tests/`): a fake crate with `csrr a0, vlenb` fails, one with a vector instruction before `vsetvli` fails, a correct one passes.

### 8. Instruction counting

- `cargo xtask count --platform riscv64-rvv --vlen <V>` (default V = 256; `--vlen all` runs 128, 256, 512 and 1024) uses the method of M13 §4 (differential K = 0 and K = 16 runs, `N = 4096`, `qemu-riscv64 -cpu rv64,v=true,vlen=<V> -plugin <libinsn.so> -d plugin`, plugin built as in M13) for our kernels and for the **own scalar fallback** as the comparison (the plan: "no established library"). Writes `benches/results/riscv64-rvv-vlen<V>.json` (instructions per sample of both builds, ratio, SNR against the f64 reference of both).
- Limits in `benches/limits.json` for the platform `riscv64-rvv`: `release` entries fail above **1.15×** the fallback (the plan's hard limit: the backend must not be worse), `dev` above 1.50×; between 1.10× and 1.15× a warning. In addition a **scaling check** for every kernel flagged `scalable` (all except `dot` C32): instructions per sample at VLEN 512 must be at most **0.67×** those at VLEN 128; the check fails otherwise (it proves that the code really uses the longer vectors).
- The report states that one vector instruction counts as one instruction in the plug-in whatever the number of elements it processes, and that QEMU emulates vectors element by element, so counts show instruction economy, not real speed; real-time targets come with real hardware.

### 9. CI (`.github/workflows/ci.yml`)

| Job | Trigger | What |
| --- | --- | --- |
| `rvv-qemu` | every push/PR (ubuntu-24.04) | `cargo xtask rvv-matrix` |
| `emu-tier2` (M15) | nightly | now also `xtask emu --target qemu-riscv64-rvv --vlen 128`, `256`, `512` (quick) and `--full` at 256 |
| `count-rvv` | PRs touching `crates/backends/sdrk-backend-rvv` or `sdrk-core`: V = 256; nightly: `--vlen all` | `cargo xtask count --platform riscv64-rvv` |

The existing `tier1` job runs `cargo xtask check` (including `check rvv`) on all three hosts; the crate is empty there, so nothing else changes.

## Test vectors

No new golden files (R8): RVV kernels are compared with the fallback and the reference through the existing golden cases (they run through the facade) and with the digest test. Seeds of the stress inputs: **500–519**.

## Tests

1. `cargo xtask rvv-matrix` passes: all workspace tests at VLEN 128/256/512/1024 with the backend active, digests equal across the four vector lengths, and the `v=false` run uses the fallback (`backend_for` = `"fallback"`) without a trap.
2. The digest test fails when a kernel is deliberately made VLEN-dependent (check once by hand with a temporary change and note the result in the decisions log; do not commit the change).
3. The quick suite passes in `targets/qemu-riscv64-rvv` at VLEN 128, 256 and 512.
4. `xtask check rvv` negative tests (§7) pass.
5. `xtask count --platform riscv64-rvv --vlen all` meets the limits and the scaling check for every kernel marked `release`.
6. Host builds on x86_64, arm64 and Windows are unaffected (crate empty); `cargo test --workspace --all-features` still passes there.

## Acceptance

R1.2 commands; `cargo xtask rvv-matrix`; `cargo xtask emu --target qemu-riscv64-rvv --vlen 256 --full`; `cargo xtask count --platform riscv64-rvv --vlen all`. **RVV backend complete.**

## Deliverables checklist

- [ ] Toolchain smoke test passed and logged
- [ ] `sdrk-backend-rvv` with nine kernels (asm functions, safe wrappers), facade feature `rvv`, `Rvv::available`
- [ ] 16-lane `dot` model documented with the exact reduction tree
- [ ] `xtask rvv-matrix`, digest test, `xtask check rvv` with negative tests
- [ ] `targets/qemu-riscv64-rvv`, `xtask emu --vlen`
- [ ] Instruction counting at all VLEN, `limits.json` entries, results, scaling check
- [ ] CI jobs, `.devcontainer/Dockerfile` additions (riscv64 targets, `gcc-riscv64-linux-gnu`)
- [ ] `docs/backends.md` (RVV section: sources, LMUL table, QEMU version, psABI citation), `docs/embedded.md`, CHANGELOG, decisions log
