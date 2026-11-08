# M54 — Kernel `code_track`: delay-locked loop

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M38](M38-symbol-sync.md), [M52](M52-despread.md) · Next: [M55](M55-pilot-equalize.md)

> **Status: planned** (written 2026-11-08). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone.

## Goal

A kernel `code_track` that keeps a direct-sequence receiver aligned to the chip clock after acquisition: a delay-locked loop with early, prompt and late correlators, a non-coherent code discriminator, a loop filter, and the sub-sample interpolator of `symbol_sync`. It consumes the chip stream and outputs one prompt (despread) symbol per code period. With the loop bandwidth set to zero and the on-time phase fixed it reduces, on the prompt arm, to `despread` (M52) bit for bit. The discriminator and loop follow the forms of GNSS-SDR's `GPS_L1_CA_DLL_PLL_Tracking`.

## Why

Acquisition (`code_search`, M53) gives a coarse code phase and Doppler; from there a tracking loop must follow the chip clock as it drifts (oscillator error, Doppler, ppm clock offset), or the correlation collapses within milliseconds. The early-minus-late discriminator measures the alignment error, and a loop filter steers an interpolator to the chip timing. This is the receive-side counterpart of `symbol_sync`, but at the chip rate and with a code: without it a long-code DSSS link cannot stay despread.

## Order

M54 depends on M38 (the `symbol_sync` interpolator and loop-filter form it reuses) and M52 (`despread`, which the prompt arm must equal at zero bandwidth). It can run in parallel with every milestone after M52; the DSSS receiver milestone needs M53 and M54 together.

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
| `crates/proto` | `sdrk-proto` | `sdrk-device` | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core, proto | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, proto, facade, core | no (std/alloc allowed) |
| `crates/io` | `sdrk-io` | core (types only) | no |
| `crates/testing` | `sdrk-testkit` | any library crate above | `no_std` by default; feature `std` adds file I/O and proptest helpers |
| `bindings/*`, `examples/*`, `targets/*`, `benches/*`, `xtask` | — | anything | no |

Additional rules:
- Within `crates/core` the rank is by crate name: `sdrk-device` (rank 0) < `sdrk-core` (rank 1); `sdrk-core → sdrk-device` is allowed, the reverse is not. All other layers are one rank per folder.
- Backends never depend on each other. Only the facade `sdrk` knows the backends.
- `sdrk-proto` (from M37 on) holds protocol logic without signal processing (bit and byte framing, checksums, packet types); it depends on `sdrk-device` only, and nothing below `crates/blocks` depends on it.
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
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 32 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.




## Scope

**In:** kernel `code_track` (`C32`) in `sdrk_core::kernels::code_track` with `CodeTrackState`, Fallback, Reference, facade delegation, contract, `docs/kernels.md`; golden cases; unit, lock, invariance, extreme tests; suite cases; `cargo xtask count`; CHANGELOG, decisions log.

**Out:** the carrier loop (that stays in `costas`, M40, placed after `code_track` as for PSK); acquisition (`code_search`, M53); the block and kind `code_track`/`dsss_demod` (later milestone; names reserved); `CQ15` (backlog); a second (data-aided) discriminator (backlog). Carrier-aiding of the code NCO (feeding the carrier Doppler into the code rate) is in as an optional input, because without it a long integration slips.

## Specification

General rules of M19 apply. The loop feeds back on itself, so as for `symbol_sync` and `costas` the arithmetic is **fixed here** and the kernel is bit-exact to its own float32 model (the float32-model rule, maintainer decision D111); `Reference` follows it step by step and the golden cases come from the exact float32 model with an f64 regression guard. The kernel is a recursion over code periods and is **not accelerated**.

### 1. Correlators and interpolator

The receiver is aligned when the prompt replica sits on the incoming chips. Three correlators per code period, at code offsets `−δ`, `0`, `+δ` chips (`δ` = `early_late_chips`, default 0.5, range 0.1…1.0; GNSS-SDR's `early_late_space_chips` default is 0.25, ours 0.5 as the DSSS default, logged):

- The input chip stream is interpolated to the loop's current chip positions with the **cubic interpolator of `symbol_sync`** (`interp(i, mu)` of the M38 contract, f32, no FMA), so a sub-sample timing offset is realised exactly as in symbol timing recovery.
- For each code period the early, prompt and late despread sums `E`, `P`, `L` are formed over the `sf·samples_per_chip` interpolated samples against the local code at the three offsets, in the `dot_complex` order of M43 (so `P` equals `despread` of the on-time samples). `samples_per_chip` ≥ 1; with `samples_per_chip = 1` the early/late offsets fall on interpolated positions between chips.

### 2. Discriminator and loop

- **Code discriminator** (non-coherent early-minus-late envelope, normalised; the GNSS-SDR BPSK form): `e = 0.5 · (|E| − |L|) / (|E| + |L|)` in f32, with `|·|` the magnitude from the `mag` kernel's rule; `e = 0` when `|E| + |L|` is 0 or non-finite (silence, NaN). The factor `0.5` is the GNSS-SDR normalisation `(y_intercept − slope·ε)/slope` for a 1-chip correlation triangle with spacing `δ`; the exact gain constant is `code_disc_gain` (default derived from `δ` as `0.5/δ`·… — the implementer fixes the constant so the discriminator slope is 1 per chip of error at `δ` = 0.5 and unit amplitude, and logs the value; a unit test measures the slope).
- **Loop filter:** the second-order filter of `symbol_sync`/`costas` in the same form, parameters `loop_bw` (0.0001…0.1 of the code rate; default 0.005) and `damping` (default 1/√2). Constants in f64: `theta = loop_bw/(damping + 1/(4 damping))`, `d = (1 + 2 damping·theta) + theta²`, `kp = ((4 damping·theta)/d)/code_disc_gain`, `ki = ((4 theta²)/d)/code_disc_gain`, rounded to f32; an integrator limit `max_rate` chips per period (default 0.05, i.e. ±5 % clock error). The NCO update per code period: `rate = clamp(flush_denormal(rate + ki·e), −max_rate, max_rate)`, `step = clamp(kp·e, −|…|, …)` (the clamp of `symbol_sync` keeping positions monotone), the chip position advances by `sf + rate + step` chips; the fractional position is kept as an integer index and an f64 `mu` exactly as `symbol_sync` does, so the result is chunk-invariant.
- **Carrier aiding (optional):** a per-period input `carrier_doppler_hz` (or 0) scales the nominal advance by `(1 + f_carrier/f_rf)`; with 0 it is off. This keeps the code and carrier Doppler consistent over long integrations (standard in GNSS tracking). Logged as the one addition beyond the GNSS-SDR BPSK DLL.
- **Output:** one prompt symbol `P` per code period (the despread symbol, handed to `costas` and the slicer downstream). `lock_level()` is a smoothed `|P|²/((|E|²+|L|²)/2)`-style metric (documented), `locked()` its threshold; `code_phase()`, `rate()`, `mu()` read the loop.

### 3. State and trait

```rust
pub struct CodeTrackState<..> { /* interpolator window, code storage, sf, samples_per_chip, δ, loop state (rate,integ,i,mu), lock */ }
impl CodeTrackState<..> {
    pub fn new(code: CodeStore, sf: u32, samples_per_chip: f64, early_late_chips: f64,
               loop_bw: f64, damping: f64, max_rate: f64, f_rf: f64) -> Result<Self>;
    pub fn gains(&self) -> CodeTrackGains;    // kp, ki, max_rate, code_disc_gain
    pub fn code_phase(&self) -> f64;  pub fn rate(&self) -> f64;  pub fn locked(&self) -> bool;
    pub fn set_initial(&mut self, code_phase_chips: f64, rate: f64);  // from code_search
    pub fn reset(&mut self);
}
pub trait CodeTrack {
    fn code_track(&self, state: &mut CodeTrackState<..>, input: &[C32], carrier_doppler_hz: &[f32], output: &mut [C32]) -> Result<Progress>;
}
```

- Parameter ranges validated in `new` (as `symbol_sync`/`costas`): `sf` 2…65536, `samples_per_chip` 1…1024, `early_late_chips` 0.1…1.0, `loop_bw` 0.0001…0.1, `damping` 0.1…4, `max_rate` 0…0.5, `f_rf` > 0 or 0 to disable aiding; non-finite or out-of-range → `InvalidParameter`.
- `carrier_doppler_hz` is either empty (aiding off) or one value per produced symbol; a length that is neither 0 nor matching the symbols produced is `LengthMismatch`.
- **Zero-bandwidth identity:** `loop_bw`’s effect is `kp = ki = 0` only at the limit; the invariant tested is: with `set_initial(phase, 0)`, `loop_bw` at its minimum, `samples_per_chip = 1`, integer alignment and `δ` such that early/late are unused, the prompt output equals `despread` (M52) of the same chips bit for bit. (A dedicated internal switch for exact zero bandwidth may be added for the test; logged.)
- Timing: `RatioKind::Variable`, nominal ratio `1/(sf·samples_per_chip)`, group delay as `symbol_sync` (the interpolator window). Consumes whole code periods; keeps the remainder in the interpolator history; chunk-invariant.

### 4. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `code_track` | `kernels::code_track::CodeTrack` | `C32` | bit-exact | the float32 model of §1–§2 (the `symbol_sync` interpolator, the `dot_complex` correlation order, the normalised E−L discriminator, the second-order loop of `costas`), each operation f32 without FMA; `Reference` follows it step by step; golden cases from the exact float32 model, an f64 loop as a regression guard (`snr_db_guard`); not accelerated (`backend_for` reports `fallback`) |

### 5. Golden vectors

New file `tests/golden/sdrk_golden/cases/code_track.py`, the float32 model of the loop (as `costas.py`/`symbol_sync.py`).

| Case | Seed | `sf` | sps | scenario |
| --- | --- | --- | --- | --- |
| `code_track/lock_sf127_40ppm` | 740 | 127 | 2 | a 127-chip code, 40 ppm clock drift, Eb/N0 10 dB; the loop follows the drift, prompt symbols recovered |
| `code_track/acq_handoff_sf1023` | 741 | 1023 | 1 | `set_initial` from an acquisition offset of 0.3 chip; the loop pulls in within N periods |
| `code_track/zero_bw_equals_despread` | 742 | 31 | 1 | min bandwidth, integer alignment; the prompt output equals `despread` (stored from `despread/…` for cross-check) |

- Stored: the code, the input, the loop parameters, the carrier-aiding array (empty or values); expected prompt symbols, the lock flag per period, the final loop state (rate bits, integrator bits, `i`, `mu`), the loop constants as IEEE bits. `snr_db_guard` from the f64 loop.

### 6. Cost

`cargo xtask count` entry `code_track_sf127` (per input chip; three correlators + the interpolator + the loop update per period). Record in `docs/performance.md`; note the loop is a recursion (not vectorised), like `symbol_sync` and `costas`.

## Tests

Seeds 740 … 749.

1. Golden cases of §5 through facade, `Fallback`, `Reference` (`sdrk-testkit/tests/golden_m54.rs`), with the f64 guard.
2. Unit tests (`sdrk-core/tests/kernels_m54.rs`): construction and range rules; the loop constants; the discriminator slope (≈ 1 per chip of offset at unit amplitude, `δ` = 0.5); lock onto a clean code with a static offset; the integrator following a constant clock drift; the carrier-aiding scaling; `set_initial`; `reset`; the length rule on `carrier_doppler_hz`.
3. Zero-bandwidth identity: the prompt output equals `despread` bit for bit in the configured case (seed 743).
4. Lock / BER sanity (`sdrk-testkit/tests/code_track_m54.rs`, seed 744): transmit `spread` → channel (`Awgn`, `ClockDrift`, `FrequencyOffset` of M38/M43) → `code_track` → the recovered symbols decode (after a `costas` and slicer) with a BER near the BPSK theory shifted by the processing gain, over enough symbols; a smoke test, not the full BER milestone.
5. Block-size invariance: 20 chunk plans at the chip level (seed 745).
6. Extremes: silence (no lock, no panic); full scale; NaN/∞ in one chip (`e = 0` that period, recovers); `samples_per_chip = 1` with `δ` = 0.1; the largest `sf`.
7. Suite cases `code_track_c32` and `extremes_m54`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo xtask count --platform x86_64` (the entry of §6).

## Deliverables checklist

- [ ] Kernel `code_track` (`C32`): `CodeTrackState`, `CodeTrackGains`, trait, Fallback, Reference, facade delegation, `backend_for` (fallback)
- [ ] Contract row and section in `docs/kernels.md` (E−L discriminator, the loop, the interpolator reuse, the `despread` identity, the GNSS-SDR citation)
- [ ] Golden cases (seeds 740–742) and the float32 Python model with the f64 guard
- [ ] Tests 1–6 (seeds 740–745); suite cases
- [ ] `cargo xtask count` entry, `docs/performance.md`
- [ ] Reserved kind names `code_track`, `code_acquire` in `docs/blocks.md`
- [ ] CHANGELOG, decisions log (the discriminator gain constant, the `δ` default, carrier aiding, the zero-bandwidth switch)

## Status

Planned. Nothing is implemented.
