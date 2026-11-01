# M42 — RDS transmit

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M41](M41-rds-rx.md), [M36](M36-fm-stereo-tx.md), [M37](M37-packet-ports.md) · Next: [M43](M43-equalizer.md)

## Goal

The FM stereo transmitter of M36 gets RDS: a block `rds_encode` (a group carousel for PS and RT plus groups from packets, encoded into the shaped biphase baseband signal, driven by the sample clock of the multiplex) and a kind `fm_stereo_rds_mod` (the stereo multiplex of M36 plus the RDS signal on 57 kHz, locked to three times the pilot). The flow `fm_stereo_rds_transmitter` and a loop test with the receiver of M41 prove it.

## Why

With RDS transmit, the library can produce a complete broadcast signal for tests and low-power experiments, and the receiver of M41 gets a second, independent test source.

## Order

M42 depends on M41 (`sdrk_proto::rds`), M36 (`stereo_encode`) and M37 (`Clocked`, packet inputs; through it M34). It can run in parallel with M43 and M44.
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

**In:** `sdrk_proto::rds::Carousel` (group generator); kernel `rds_shape` (f32) and kernel extension `stereo_encode_rds` in `sdrk-core`; blocks `RdsEncodeBlock` and `FmStereoRdsModBlock`; kinds `rds_encode` and `fm_stereo_rds_mod`; the flow `examples/flows/fm_stereo_rds_transmitter.json` with a YAML twin; tests; `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, `CHANGELOG.md`.

**Out:** text parameters in the parameter system (backlog: PS and RT arrive as packets, §2), clock-time and AF groups (backlog; they can be sent as raw groups), RDS2.

## Specification

### 1. `sdrk_proto::rds::Carousel`

`Carousel::new(pi: u16, pty: u8, tp: bool)`; `set_ps(&[u8; 8])`, `set_rt(&[u8])` (≤ 64 bytes, padded with `0x0D` and spaces as the standard says, toggles the A/B flag on change), `push_raw(Group)`, `next_group() -> [u16; 4]`: raw groups first (FIFO, at most 16 queued), else alternately one 0A segment (cycling over the 4 PS segments) and one 2A segment (cycling over the 16 RT segments; none while RT is empty). Block B of 0A carries TP, PTY, TA = 0, M/S = 1, DI per segment = 0; block C of 0A is `0xE0E0` (the "no alternative frequencies" filler; record in the decisions log). Test against `RdsState` of M41: every group of the carousel parses back to the text that was set.

### 2. `rds_encode`

`RdsEncodeBlock`: ports `Clocked<C32, Packets>` (port a: any stream at the multiplex rate, here the stereo audio after the up-sampling `resample` node; its values are ignored), Out `f32` at the clock rate, ratio 1, **delay reported as 0** (the data timing relative to the audio does not matter, and a fractional delay would block the alignment of M34; record in D-log).

Packets on port b control the encoder by their first byte: `0x01` + 8 bytes PS; `0x02` + up to 64 bytes RT; `0x03` + 8 bytes one raw group (4 big-endian words). Other packets are counted and ignored (`stats`). Parameters: `pi` (I64, build, required; 0 … 65535), `pty` (I64, build, 0; 0 … 31), `tp` (Bool, build, false).

Per data bit (1187.5 bit/s; positions in samples are `n / fs` in f64): the carousel gives 4 blocks; each block is `info·2^10 + checkword` (26 bits, most significant bit first); differential encoding `t_k = t_{k−1} XOR d_k` (`t_{−1} = 0`); amplitude `a_k = 2·t_k − 1`. Output with the kernel `rds_shape` (§3).

### 3. Kernel `rds_shape`

The shaped biphase pulse `p(t)` is a table of 1024 f32 values over `4·Tb` (256 per bit, `t` from `−2·Tb`), computed in `new` (f64): `h(t)` = the impulse response of `H(f) = cos(π·f·Tb/4)` for `|f| ≤ 2/Tb`, else 0 (inverse FFT of 8192 points, then sampled), `p(t) = h(t + Tb/4) − h(t − Tb/4)`, normalized to a peak of 1. Output sample `n`: `b[n] = Σ a_k·p(t_n − k·Tb)` over the (at most 4) bits whose pulse covers `t_n`, with linear interpolation in the table at the fractional position (f32, `k` ascending). Bit-exact against the float32 model; the f64 table is part of the golden case.

```rust
pub struct RdsShapeState<Tab: AsRef<[f32]>> { /* table, time position (f64), bit FIFO of 4 */ }
pub trait RdsShape { fn rds_shape<Tab>(&self, s: &mut RdsShapeState<Tab>, bits: &mut dyn FnMut() -> i8, output: &mut [f32]) -> Result<()>; }
```

(`bits` is called whenever a new bit enters the window; the block passes the carousel's next bit.)

### 4. `fm_stereo_rds_mod`

`FmStereoRdsModBlock`: ports `Two<C32, f32>` (stereo audio, RDS baseband), Out `f32`. `stereo_encode_rds`: as `stereo_encode` (M36 §2) plus `+ rds·b·s3` with `s3 = sin(3φ)` from the tables at the phase `φ.wrapping_mul(3)`, added last. Parameters as `fm_stereo_mod` plus `rds` (None, **runtime**, 0.04); the default `main` is 0.87 here, so `main + pilot + rds ≤ 1`. Cost 14.

### 5. Flow

`fm_stereo_rds_transmitter` (+ YAML): `audio` C32 at 48 kHz → `pre` = `preemph_limit` → `up` = `resample` (192000) → edges to `mpx` port a and to `rds` port a (the clock); `text` packets input → `rds` port b; `rds` = `rds_encode` (pi 0xD123 as 53539) → `mpx` port b; `mpx` = `fm_stereo_rds_mod` → `mod` = `fm_mod` (75000) → `duc` (interpolation 10, 600000) → `rf` `cs8`, pull mode.

## Tests

1. Golden: `rds_shape/table_and_bits_f32_16384` (seed 580, float32 model); `stereo_encode_rds/c32_16384` (seed 581).
2. Spectrum: the RDS component of the multiplex lies within 57 kHz ± 2.4 kHz (outside at least 40 dB below its peak), and at exactly 57 kHz its level is at least 30 dB below that peak (the biphase coding has no carrier; seed 582).
3. **Loop (`e2e_rds_loop`):** the flow (without `duc`, at 192 kHz, then `resample` to 240 kHz) → `fm_rds_receiver` chain from `quad_demod` on: PS and RT set by packets decode within 2 s and 6 s; 100 % valid groups at 40 dB SNR, ≥ 99 % at 25 dB (seeds 583, 584); a raw group sent by packet arrives unchanged; the stereo separation of M36 E2E-STEREO-TX still holds (30 dB).
4. Cross-check: the Python reference modulator (M41 §5) and `rds_encode` give the same RDS baseband within 1e-5 for the same groups (seed 585).
5. Chunk invariance with 20 plans (seed 586).

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-proto`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`.

## Deliverables checklist

- [ ] `Carousel` in `sdrk_proto::rds`
- [ ] Kernels `rds_shape` and `stereo_encode_rds` with reference, fallback, delegation, contracts
- [ ] Blocks and kinds `rds_encode` (`Clocked`), `fm_stereo_rds_mod` (`Two`)
- [ ] Flow `fm_stereo_rds_transmitter` with YAML twin
- [ ] Tests (seeds 580–586)
- [ ] `docs/kernels.md`, `docs/blocks.md`, `docs/pipeline.md`, `examples/README.md`, CHANGELOG, decisions log
