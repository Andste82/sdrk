# M35 — AM and SSB transmit: `am_mod` and `ssb_mod`

> Plan: [PLAN.md › Phase 9](../PLAN.md#phase-9--stabilization-and-extensions) · Depends on: [M29](M29-primitive-blocks.md), [M30](M30-two-port-blocks.md) · Next: [M36](M36-fm-stereo-tx.md)

## Goal

The transmit counterparts of `am_demod` (M20) and `ssb_demod` (M21): a kernel and block `am_mod` (audio → AM baseband with carrier), a block `ssb_mod` (audio → USB or LSB baseband, the Weaver method mirrored from `ssb_demod`), pipeline kinds, two example transmit flows in pull mode and loop tests TX → RX.

## Why

So far only FM can be transmitted (`fm_mod`, M08). AM and SSB complete the analog modes for amateur radio and test setups, and the loop tests prove that the receivers of M20/M21 and the new modulators agree on levels, sidebands and tuning conventions.

## Order

M35 depends on M29 (`rotate`) and M30 (`to_complex`). It can run in parallel with M31–M34 and M36.
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

**In:** kernel `am_mod` (f32 → C32, Q15 → CQ15) in `sdrk-core` with Fallback, Reference, facade delegation and contract; golden cases; blocks `AmModBlock<T>` and `SsbModBlock<T>` with descriptions; pipeline kinds `am_mod` and `ssb_mod`; schema; flows `examples/flows/am_transmitter.json` and `ssb_transmitter.json` with YAML twins; loop tests; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** DSB with suppressed carrier as its own mode (backlog), an audio compressor or speech processor (the limiter `preemph_limit` of M25 is for FM), SSB by the phasing method (Hilbert filter), CW keying with shaped edges (backlog).

## Specification

General rules of M19 apply (exact order, no FMA, Q15 in `i64`, R7).

### 1. Kernel `am_mod`

```rust
pub trait AmMod<T: ComplexSample> {
    /// out[n] = (c·(1 + m·x[n]), 0) with x clamped to [−1, 1].
    fn am_mod(&self, carrier: f32, depth: f32, input: &[T::Real], output: &mut [T]) -> Result<()>;
}
```

- C32: `x' = clamp(x, −1, 1)` (NaN → 0), `t = depth·x'`, `u = 1 + t`, `re = carrier·u`, `im = 0`.
- CQ15: `c_q = round(carrier·32768)` limited to 32767, `m_q = round(depth·32768)` limited to 32767; `t = round_shift(m_q·x, 15)`, `u = 32768 + t` (Q15, 0 … 65535), `re = sat16(round_shift(c_q·u, 15))`, `im = 0`; all in `i64`.
- `carrier` 0 … 1 and `depth` 0 … 1, else `InvalidParameter`. Unequal lengths give `LengthMismatch`. Stateless, ratio 1, delay 0; bit-exact class.

### 2. Block `am_mod`

`AmModBlock<T>`: f32 → C32, Q15 → CQ15. Parameters: `carrier` (None, build, 0.5; 0 … 1: the carrier amplitude, so that the peak is `carrier·(1 + depth) ≤ 1` at the defaults), `depth` (None, **runtime**, 0.8; 0 … 1: the modulation index). Kind `am_mod`, cost 2. `docs/blocks.md` chain position: a band-limiting low-pass on the audio (`biquad` low-pass 4500 Hz for broadcast AM) before `am_mod`, then `duc`.

### 3. Block `ssb_mod`

`SsbModBlock<T>`: f32 → C32, Q15 → CQ15, ratio 1. It needs no kernel of its own and mirrors `ssb_demod` (docs/blocks.md › `ssb_demod`): with `c = (low_hz + high_hz) / 2`, `b = (high_hz − low_hz) / 2` and `s = +1` for USB, `−1` for LSB, per piece of 256 samples:

1. `interleave(re = x, None)`: the real audio as a complex signal.
2. `rotate` by `−c` (NCO 1 at the audio rate): the band `[low_hz, high_hz]` moves to `[−b, b]`.
3. `fir` complex low-pass with cutoff `b` (taps from `kaiserord` and a Kaiser `firwin` with `transition_hz` and `attenuation_db`, then **multiplied by 2** before quantization, so a real tone of amplitude A comes out with amplitude A; quantized as in the `fir` block).
4. `rotate` by `+s·c` (NCO 2): the band moves to `[low_hz, high_hz]` (USB) or `[−high_hz, −low_hz]` (LSB).

Parameters: `sideband` (Choice `usb`/`lsb`, **runtime**, `usb`; flips the sign of NCO 2 only, both phases continue), `low_hz` (Hz, build, 300), `high_hz` (Hz, build, 3000), `transition_hz` (Hz, build, 200), `attenuation_db` (dB, build, 60), `max_taps` (I64, build, 1023). Validation as `ssb_demod`: `0 ≤ low_hz < high_hz`, `high_hz + transition_hz / 2 ≤ fs / 2`, else `OutOfRange`. Timing: ratio 1, delay `(taps − 1) / 2`. Kind `ssb_mod`, cost `8 + 4·taps`. Each kernel is called once per piece, never per sample.

A tone at audio frequency `f` (inside the band) comes out at `+f` (USB) or `−f` (LSB) with the input amplitude; the opposite sideband is suppressed by the filter. The name convention matches `ssb_demod`: `ssb_mod` with `usb` followed by `ssb_demod` with `usb` returns the audio.

### 4. Golden cases (`am_mod.py`, `ssb_mod.py`)

| Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- |
| `am_mod/c32_4096`, `am_mod/cq15_4096` | 480 | a 1 kHz tone 0.7 plus noise σ 0.2 at 48 kHz, 32 values beyond ±1 and NaN (C32); carrier 0.5, depth 0.8 | float32 / integer model | bit-exact |
| `blocks/ssb_mod_usb_c32`, `blocks/ssb_mod_lsb_c32` | 481 | two tones 700 Hz 0.3 and 1900 Hz 0.2 at 48 kHz, defaults | f64 chain (`upfirdn`, `exp`) | SNR ≥ 90 dB |
| `blocks/ssb_mod_usb_cq15` | 482 | the input above quantized | integer chain | bit-exact |

## Tests

1. Golden tests whole and chunked (20 chunk plans for `ssb_mod`, seed 483).
2. Extremes (`extremes_m35`): silence, full scale, NaN/∞, Q15 corners; `depth` 0 gives the constant carrier; `depth` 1 with a full-scale tone gives an envelope that touches 0.
3. **Loop AM:** `am_mod` (48 kHz, 1 kHz tone 0.5, depth 0.8) → `am_demod` (defaults) gives the 1 kHz tone with SINAD ≥ 40 dB (f32) and ≥ 30 dB (Q15).
4. **Loop SSB:** `ssb_mod` → `ssb_demod` with the same sideband and band: a 1 kHz tone 0.5 comes back with amplitude within 0.5 dB, frequency exact (FFT bin); with the opposite sideband in the demodulator the level is at least 50 dB lower. The spectrum of the `ssb_mod` output has the wanted tone at `+1 kHz` (USB) and at most −50 dB at `−1 kHz`.
5. Sideband switch at runtime: phase-continuous (no step larger than the signal amplitude in the output), test with seed 484.
6. Flows: `am_transmitter` (`audio` f32 48 kHz → `lpf` = `biquad` low-pass 4500 → `mod` = `am_mod` → `duc` (interpolation 50, frequency 600000) → `rf` `cs8`, pull mode) and `ssb_transmitter` (`audio` → `mod` = `ssb_mod` → `duc` (interpolation 50, frequency 0) → `rf` `cs8`): validate, build, pull 10^6 bytes; the pull output equals a push run of the same audio.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] Kernel `am_mod` (C32, CQ15) with reference, fallback, delegation, contract
- [ ] Blocks `am_mod`, `ssb_mod`; kinds, costs, schema
- [ ] Golden cases (seeds 480–482), extremes, chunk invariance (483), sideband switch (484)
- [ ] Loop tests AM and SSB
- [ ] Flows `am_transmitter`, `ssb_transmitter` with YAML twins
- [ ] `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/backends.md` (`am_mod` accelerated where cheap, else "Not accelerated"), `examples/README.md`, CHANGELOG, decisions log
