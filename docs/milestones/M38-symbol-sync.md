# M38 — Symbol timing recovery, slicer, test channel and BER meter

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M29](M29-primitive-blocks.md), [M37](M37-packet-ports.md) · Next: [M39](M39-fsk.md)

## Goal

The shared building blocks of every digital receiver, before any modulation: a kernel and block `symbol_sync` (symbol clock recovery with the Gardner or the Mueller–Müller detector and a cubic interpolator), a block `slice` (symbols → soft bits for BPSK, QPSK and real levels), a PRBS generator in `sdrk-proto`, and in the testkit a channel model (noise, frequency offset, clock drift) and a BER meter. The tests prove that `symbol_sync` reaches the theoretical bit error rate of BPSK within a small margin.

## Why

FSK, PSK, QAM and RDS (M39–M44) all need the symbol clock, and all their tests need a channel and a BER measurement. Building and testing these once, without a modulation on top, keeps the following milestones small and makes their tests comparable.

## Order

M38 depends on M29 (`delay`, `decimate` for the tests) and M37 (soft bits feed `frame_decode`; PRBS lives in `sdrk-proto`). It can run in parallel with M35 and M36.
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
6. Each case (JSON + all its `.bin` arrays) stays at or below 512 KiB; all vectors together at or below 16 MiB. Prefer N ≤ 4096 samples; use longer inputs only when the kernel needs them (long filters, decimation).
7. Tests that use `sdrk-testkit` live in integration tests (`<crate>/tests/*.rs`) or in `crates/testing/sdrk-testkit/tests/`, never in `#[cfg(test)]` unit modules of a crate the testkit itself depends on (a dev-dependency cycle would compile two copies of that crate and produce confusing trait errors).

### R9 — Definition of done (every milestone)

- All commands in R1.2 and the milestone's *Acceptance* section pass locally.
- Every new public item is documented with an example; `docs/` files named in the milestone are updated.
- New kernels have: a reference, a fallback, a facade delegation, golden cases, a block-size-invariance test (if stateful), extreme-input tests (silence, full scale, NaN/∞ for f32 inputs) and an entry in the testkit suite.
- `CHANGELOG.md` has an entry under "Unreleased" for every user-visible change.
- `docs/decisions-log.md` lists every judgement call made during the milestone.


## Scope

**In:** kernel `symbol_sync` (C32 and f32) in `sdrk-core` with Fallback, Reference, facade delegation, contract; block `SymbolSyncBlock<T>` and kind `symbol_sync`; block `SliceBlock` and kind `slice` (§3, no new kernel); `sdrk_proto::prbs` (PRBS-9, PRBS-15); in `sdrk-testkit`: `channel::{Awgn, FrequencyOffset, ClockDrift, ChannelChain}` and `ber::BerMeter`; golden cases; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `docs/testing.md`, `docs/backends.md`, `CHANGELOG.md`.

**Out:** carrier recovery (`costas`, M40), pulse-shape design (RRC, M40), slicing of 8PSK and QAM (M40.1, M44), equalizers (M43), Q15 variants (backlog: the digital blocks are C32/f32 only, D-number in the decisions log), multipath channels (M43).

## Specification

General rules of M19 apply. `symbol_sync` is a feedback loop: its golden cases come from an exact float32 model (the `squelch` rule) with an f64 regression guard.

### 1. Kernel `symbol_sync`

```rust
pub enum Ted { Gardner, MuellerMuller }
pub struct SymbolSyncState<T> { /* history of 4 samples, t (f64), omega (f64), omega_nominal, integrator, last symbol and decision */ }
impl<T> SymbolSyncState<T> {
    /// sps = fs / symbol_rate ≥ 2 (not necessarily an integer); loop_bw 0.0001 … 0.1 (normalized to the
    /// symbol rate), damping 0.1 … 4 (default 1/√2), max_deviation 0 … 0.05 (fraction of sps), ted_gain > 0.
    pub fn new(sps: f64, ted: Ted, loop_bw: f64, damping: f64, max_deviation: f64, ted_gain: f64) -> Result<Self>;
    pub fn reset(&mut self);
    pub fn sps_estimate(&self) -> f64;   // current omega
}
pub trait SymbolSync<T> { fn symbol_sync(&self, s: &mut SymbolSyncState<T>, input: &[T], output: &mut [T]) -> Result<Progress>; }
```

Loop constants (f64, in `new`): `θ = loop_bw / (damping + 1/(4·damping))`, `d = 1 + 2·damping·θ + θ²`, `kp = (4·damping·θ / d) / ted_gain`, `ki = (4·θ² / d) / ted_gain`.

Interpolation at the position `t = i + μ` (integer `i`, `0 ≤ μ < 1`) from `x[i−1], x[i], x[i+1], x[i+2]` (cubic Lagrange, per component in f32, `μ` cast to f32):

```
c0 = x[i]
c1 = −x[i−1]/3 − x[i]/2 + x[i+1] − x[i+2]/6
c2 = x[i−1]/2 − x[i] + x[i+1]/2
c3 = −x[i−1]/6 + x[i]/2 − x[i+1]/2 + x[i+2]/6
y  = ((c3·μ + c2)·μ + c1)·μ + c0
```

The divisions are multiplications by the f32 constants `1/3`, `1/2`, `1/6` rounded once; the terms of each `c` are summed from left to right.

Per strobe (while `floor(t) + 2` is an available input index and the output has room):

1. `y = interp(t)` (the on-time sample); Gardner also `m = interp(t − omega/2)` (the mid sample).
2. Error `e` (f32, then cast to f64): Gardner real `(y_prev − y)·m`, complex `re((y_prev − y)·conj(m))`; Mueller–Müller with decisions `d = sign(y)` (complex: per component), real `d_prev·y − d·y_prev`, complex `re(conj(d_prev)·y − conj(d)·y_prev)`. The first strobe after construction or `reset` uses `e = 0`.
3. Output `y`; `integ = clamp(integ + ki·e, −max_deviation·omega_nominal, +max_deviation·omega_nominal)`; `omega = omega_nominal + integ`; `t = t + omega + kp·e`.

`t` starts at 1 (the first full interpolation window). Consumed input is subtracted from `t` at the end of a call, so `t` stays small; the call consumes all input except the last 3 samples, which stay in the history. Timing: `RatioKind::Variable` with the nominal ratio `1 / sps` (exact: `symbol_rate` as a fraction with denominator 1000, see §2), delay 1 input sample.

### 2. Block and kind `symbol_sync`

`SymbolSyncBlock<T>`, C32 → C32, f32 → f32. Parameters: `symbol_rate_hz` (Hz, build, required; rounded to 1/1000 Hz, so 1187.5 and 2375 are exact), `ted` (Choice `gardner`/`mm`, build, `gardner`), `loop_bw` (None, build, 0.01), `damping` (None, build, 0.7071), `max_deviation` (None, build, 0.005), `ted_gain` (None, build, 1.0). `rate_unsupported` when `fs / symbol_rate_hz < 2`. Kind with cost `2 + 24 / sps` (two interpolations per symbol). `docs/blocks.md`: chain position after the matched filter; the input amplitude should be about 1 (an `agc` before it), because the loop gain is proportional to the signal power (`ted_gain` corrects other levels).

### 3. Block and kind `slice`

`SliceBlock`: symbols → soft bits (`f32`), no kernel; a stateless per-symbol map in Rust with the order given. Convention for the whole project: **a soft bit `> 0` means bit 1**, its magnitude is the confidence (record in `docs/blocks.md` and the decisions log).

| `constellation` | Input | Bits per symbol | Soft bits (`s` = `scale`) |
| --- | --- | --- | --- |
| `real` | f32 | 1 | `s·x` |
| `bpsk` | C32 | 1 | `s·re` |
| `qpsk` | C32 | 2 | `s·re`, `s·im` (Gray: bit 0 from `re`, bit 1 from `im`) |

Parameter `scale` (None, **runtime**, 1.0; a receiver sets `2/σ²` for LLRs). Output ratio = bits per symbol, delay 0, cost 1 per input sample. M40.1 and M44 add constellations to this table.

### 4. `sdrk_proto::prbs`

`Prbs::new(Poly::Prbs9 | Poly::Prbs15, seed)`: Fibonacci LFSR, `x^9 + x^5 + 1` and `x^15 + x^14 + 1`, output bit = the feedback bit, `next_bit() -> u8`; `seed` non-zero (`InvalidParameter`). Tested against the first 32 bits of each sequence with seed all-ones (listed in the test).

### 5. Testkit

- `channel::Awgn::new(sigma, seed)`, `Awgn::from_ebn0(ebn0_db, bits_per_symbol, sps, signal_power, seed)` (noise per complex sample with `σ² = signal_power·sps / (2·bits_per_symbol·10^(ebn0/10))` per component); `FrequencyOffset::new(offset_hz, fs)` (f64 phase); `ClockDrift::new(ppm)` (resampling by `1 + ppm·1e-6` with a windowed-sinc interpolator in f64, 32 taps per side); `ChannelChain` applies a list in order. All deterministic with a seed (`sdrk_testkit::rng::Pcg32`); they go into the existing `channel.rs` next to `Loopback`.
- `ber::BerMeter::new(Poly, seed)`: takes hard bits, synchronizes on the PRBS (first 32 correct bits in a row), then counts `bits` and `errors`; `ber()`, and `confidence_95()` (Wilson interval). Re-synchronizes after 64 errors in 128 bits (counts a `slip`).
- `signals::psk_test(sps, rolloff, bits)`: BPSK/QPSK symbols shaped with a raised-cosine pulse (f64 testkit design; the library RRC comes in M40) for the tests of this milestone.

### 6. Golden cases (`symbol_sync.py`)

| Case | Seed | Construction | Expected | Tolerance |
| --- | --- | --- | --- | --- |
| `symbol_sync/gardner_c32_sps4p1_16384` | 490 | BPSK (raised cosine 0.35), sps 4.1, 40 ppm drift, Eb/N0 20 dB | float32 model | bit-exact |
| `symbol_sync/gardner_f32_sps8_8192` | 491 | real ±1 levels (2-FSK discriminator output shape: boxcar-filtered), sps 8 | float32 model | bit-exact |
| `symbol_sync/mm_c32_sps2_8192` | 492 | QPSK, sps 2, timing offset 0.37 | float32 model | bit-exact |

## Tests

1. Golden tests whole and chunked (20 chunk plans, seed 493); f64 guard.
2. **Lock:** for sps 2, 4, 4.1 and 10.1, a start offset of 0.5 symbols and drift ±100 ppm, Gardner reaches a timing error below 0.05 symbols after 500 symbols (measured against the known symbol positions of the generator; seed 494).
3. **BER BPSK:** PRBS-15 BPSK, sps 4.1, raised cosine 0.35 with a matched filter in the test, 50 ppm drift, `symbol_sync` → `slice bpsk` → `BerMeter`: at Eb/N0 = 6 dB and 8 dB the BER is within a factor 1.5 of `Q(√(2·Eb/N0))` over 10^6 bits (seed 495). Mueller–Müller at sps 2 within a factor 2 (seed 496).
4. `slice`: tables of §3 for corner values; QPSK bit order.
5. PRBS sequences; `BerMeter` with injected errors (exact count) and with a slip (seed 497).
6. Testkit channel: `Awgn::from_ebn0` gives the measured Eb/N0 within 0.1 dB over 10^6 samples (seed 498); `FrequencyOffset` and `ClockDrift` against analytic tones (seed 499).
7. Pipeline: a flow `bpsk_sync_test` in `crates/pipeline/sdrk-pipeline/tests/flows/` (not an example) with `symbol_sync` → `slice` → `frame_decode` decodes 100 frames sent with `frame_encode` and a test-side raised-cosine modulator.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask emu --target qemu-cortex-m4` (quick profile).

## Deliverables checklist

- [ ] Kernel `symbol_sync` (C32, f32) with reference, fallback, delegation, contract
- [ ] Blocks and kinds `symbol_sync`, `slice`; schema
- [ ] `sdrk_proto::prbs`
- [ ] Testkit channel model, BER meter, PSK test signals
- [ ] Golden cases (seeds 490–492) and tests (seeds 493–499)
- [ ] `docs/kernels.md`, `docs/blocks.md` (soft-bit convention), `docs/pipeline.md`, `docs/testing.md`, `docs/backends.md` ("Not accelerated"), CHANGELOG, decisions log
