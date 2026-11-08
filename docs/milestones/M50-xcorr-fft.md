# M50 — Kernel `xcorr_fft`: fast convolution / long correlator

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M07](M07-fft.md), [M43](M43-equalizer.md), [M49](M49-scramble.md) (the PRBS codes of the block) · Next: [M51](M51-spread.md)

> **Status: planned** (written 2026-11-08). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone. This is the "FFT-based fast convolution (overlap-save) as a block" item of the backlog, built now as a kernel because the sync-word correlator (`sync_detect`) and the code-phase search (`code_search`, M53) both need it.

## Goal

A kernel `xcorr_fft` that convolves a `C32` stream with a fixed complex filter of up to a few thousand taps by the overlap-save method, using the `fft` kernel of the backend, at a cost per output sample that grows with `log` of the filter length instead of linearly. It is the engine of a long matched filter: a preamble or spreading-code correlator whose reference is far too long for the time-domain `dot_complex` of `equalize`. The taps are given conjugated-and-reversed by the caller when a correlation (not a convolution) is wanted, so the kernel itself is a pure linear convolution.

## Why

The sample-level sync-word correlator named in the backlog, the acquisition of a DSSS code (hundreds to thousands of chips) and any long matched filter need `sum_k x[n+k] · conj(h[k])`. Done directly this is `L` complex multiply-accumulates per output sample; a 1023-chip code at a few samples per chip is thousands of taps, and the time-domain cost is prohibitive. Overlap-save computes one block of outputs with two FFTs and a spectral product, so the cost per output sample is `O(log N)`. GNU Radio (`gr::filter::kernel::fft_filter_ccc`) and liquid-dsp (`fftfilt`) both build their long filters this way; this kernel is the heap-free, bit-reproducible version for SDRk.

## Order

M50 depends on M07 (the `fft` kernel and `FftPlan`) and on M43 (the `dot_complex` kernel, used only by the reference as the direct convolution to compare against). It is independent of the other kernels of the series and can run in parallel with every milestone after M43.

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

**In:** kernel `xcorr_fft` (`C32`) in `sdrk_core::kernels::xcorr` with `XcorrFftState`, Fallback (the overlap-save loop over the backend `fft`), Reference (the direct convolution), facade delegation, contract row and section in `docs/kernels.md`; golden cases; unit, invariance and extreme tests; suite cases; `cargo xtask count`; CHANGELOG, decisions log; the block `CorrelateBlock` and the pipeline kind `correlate` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the block and kind `sync_detect` and its peak search / threshold (later milestone; the name `sync_detect` stays reserved); `CQ15` (backlog, the FFT dynamic range needs care); a frequency-offset search (that is `code_search`, M53, which calls this kernel per Doppler bin).

## Specification

General rules of M19 apply.

### 1. Method (overlap-save)

The kernel computes the linear convolution `y[n] = sum_{k=0}^{L-1} h[k] · x[n-k]` of the input stream with the fixed taps `h` (length `L`), streaming and chunk-invariant. Overlap-save with an `N`-point FFT (`N` a power of two, `16 ≤ N ≤ 65536`, `N ≥ 2L`):

1. **Setup (`new`).** `H[·] = FFT_N{ h padded to N with zeros }`, stored once (caller storage, `N` complex values). The block size is `B = N − L + 1` output samples per transform. The state keeps the last `L − 1` input samples as history (zero at the start).
2. **Per transform block.** Form the length-`N` window `w = [ history (L−1 samples) | next B input samples ]`; `W = FFT_N{w}`; `Y = W ⊙ H` (element-wise complex product, `Y[k] = W[k]·H[k]`); `y_block = IFFT_N{Y}`; the valid outputs are `y_block[L−1 .. N−1]` (the first `L−1` are the wrapped, invalid part that overlap-save discards). Advance the history to the last `L−1` samples of `w`.
3. A call consumes whole blocks of `B` input samples and keeps the remainder in the history buffer; `Progress` counts input consumed and output produced. `min_input` is `B`. When fewer than `B` new samples are available the call produces nothing and consumes nothing (the caller resubmits). The result is independent of how the input is split into calls (same block boundaries always, because `B` is fixed).

**Correlation vs convolution.** The kernel convolves. To correlate a stream with a reference `r` (a sync word, a spreading code), the caller passes `h[k] = conj(r[L−1−k])` (reversed and conjugated); then `y[n] = sum_k conj(r[k]) · x[n−L+1+k]`, the sliding correlation, and the output sample `n` aligns the reference with the input window ending at `n`. The kernel does not reverse or conjugate; the block of the later milestone does (one line) and documents the alignment. This keeps the kernel a plain convolution and reuses it for pulse shaping as well.

**Scaling.** The forward `C32` FFT has `exponent` 0 and the inverse divides by `N` (M07), so `IFFT_N{FFT_N{w} ⊙ FFT_N{h}}` is the convolution with no extra factor: the kernel applies no scaling of its own. (GNU Radio folds `1/N` into its taps and liquid-dsp scales by `1/(2n)` because their IFFTs are unnormalised; ours is normalised, so no tap scaling is needed. Noted in the decisions log as the reason the taps are not pre-scaled.)

### 2. State and trait

```rust
pub struct XcorrFftState<T, S, Sh, Sc> { /* H: S (N complex), history: Sh (L-1), scratch: Sc (N), taps_len: usize, fft: FftPlan<..> */ }
impl XcorrFftState<..> {
    /// `taps`: the L filter taps (already reversed/conjugated for a correlation). `fft` is a
    /// plan of size N, a power of two, 16..=65536, with N >= 2*L. Fills H from the taps.
    pub fn new(taps: &[T], h_store: S, history: Sh, scratch: Sc, fft: FftPlan<..>) -> Result<Self>;
    pub fn taps_len(&self) -> usize;
    pub fn block_size(&self) -> usize;  // B = N - L + 1
    pub fn reset(&mut self);            // history to zero
}
pub trait XcorrFft<T> {
    fn xcorr_fft(&self, state: &mut XcorrFftState<..>, input: &[T], output: &mut [T]) -> Result<Progress>;
}
```

- `L = 0`, `L > N/2` (so `N < 2L`), an `N` outside the FFT range or not a power of two, or storage of the wrong length give `InvalidParameter` / `LengthMismatch` in `new`.
- Timing: ratio `1/1`, group delay `L − 1` input samples (the correlation peak of a reference aligned at input index `m` appears at output index `m`; the block documents it). `min_input` `B`.
- Storage: `H` (`N` complex), the history (`L − 1`), the scratch (`N`, the FFT works in place on the window copy; the real-FFT-style scratch of M07 is not needed because this is a complex FFT), and the FFT plan's twiddles. `storage_bytes` exact.

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `xcorr_fft` | `kernels::xcorr::XcorrFft` | `C32` | tolerance | SNR ≥ 100 dB vs. the f64 direct convolution of the reference (the `fft` kernel is tolerance; the error is that of one forward FFT, a spectral product and one inverse FFT); the fallback, the facade and (when it exists) the SIMD path agree within the `fft` backend tolerance; chunk invariance is exact in the block structure (identical block boundaries for every chunking), the output values within the `fft` tolerance across chunkings because the same FFTs run |

**Note on bit-exactness.** Because `fft` is a tolerance kernel and the backend FFT may use another twiddle source than the fallback (M07), `xcorr_fft` is tolerance, not bit-exact, and its chunk-invariance test compares within the stored SNR floor (like `channelize`), not with `==`. `Reference` computes `y[n] = sum_k h[k]·x[n−k]` directly in f64 (reusing the idea of `dot_complex` but in f64, O(N·L), size-limited like the FFT reference: the reference processes at most a few thousand output samples per case).

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/xcorr.py`. Expected values: the f64 direct convolution of the stored taps and input (`numpy.convolve(x, h)[L-1 : L-1+rows]`), independent of the FFT path.

| Case | Seed | `L` | `N` | Input | Note |
| --- | --- | --- | --- | --- | --- |
| `xcorr_fft/code_127_n512` | 700 | 127 | 512 | a 127-chip ±1 code embedded in noise, the taps = conj-reversed code | correlation peak at the embedding offset |
| `xcorr_fft/code_1023_n4096` | 701 | 1023 | 4096 | a 1023-chip ±1 code in noise at −10 dB chip SNR | the peak stands out of the noise floor by ≈ 10·log10(1023) dB |
| `xcorr_fft/rrc_65_n256` | 702 | 65 | 256 | a QPSK signal, the taps = an RRC pulse | a matched filter (real taps as complex) |
| `xcorr_fft/rand_300_n1024` | 703 | 300 | 1024 | complex Gaussian input and complex Gaussian taps | general convolution, no structure |

- Tolerances in the JSON: `tolerance` with `snr_db_min` 100 (against the f64 convolution), `skip` the first `L − 1` samples.
- The peak-standout cases (700, 701) also store the expected peak index and the ratio of the peak magnitude to the median magnitude as information (the block's job later, but a cheap sanity check here).

### 5. Cost

`cargo xtask count` entries `xcorr_fft_n512` and `xcorr_fft_n4096` (per output sample, dominated by the two FFTs over `B` outputs). Compare in `docs/performance.md` against the time-domain cost `L` complex MACs per output (the break-even `L` where overlap-save wins is recorded) and, as information, against GNU Radio's `fft_filter_ccc` and liquid-dsp's `fftfilt` instruction counts if a count harness for them exists; otherwise cite their method only (both overlap-add; ours overlap-save — noted).

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies once a SIMD path exists.

### 6. Block and kind `correlate`

`CorrelateBlock` (`C32` → `C32`) around the kernel `xcorr_fft`: the sliding correlation of the input with a reference `r` of length `L`, `y[n] = sum_k conj(r[k]) · x[n − L + 1 + k]`. The block builds the taps `h[k] = conj(r[L − 1 − k])` once (§1) and chooses the FFT size. The reference is a PRBS code (`sdrk_proto::prbs`, chip `b` → `1 − 2b`, imaginary part 0) with every chip repeated `samples_per_chip` times; `CorrelateBlock::with_reference` takes any `C32` reference from the caller (Rust API only, the pipeline has no list parameters).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `correlate` | `CorrelateBlock` | C32 → C32 | `code` (Choice, build, `prbs7`; `prbs5` … `prbs15`), `code_seed` (I64, build, 1; as `key_seed` of `scramble`), `samples_per_chip` (I64, build, 1; 1 … 16), `fft_size` (I64, build, 0 = automatic; else a power of two, 16 … 65536, at least `2 L`) | `xcorr_fft`; automatic size: the smallest power of two ≥ `4 L`, at least 16; `L` above 32768 or a manual size below `2 L` is `invalid_value` at `params.fft_size` with the required size in the message |

- `CorrelateConfig`. Storage: `H` (`N`), the history (`L − 1`), the scratch (`N`) and the twiddles of the plan; exact. Timing ratio 1, delay `L − 1` (from the kernel state), `min_input` `B = N − L + 1`; the output is tolerance class (the chunk tests compare within the stored SNR, as `channelize`).
- Pipeline kind `correlate` (`c32` only), cost `(5 N log2(N) + 4 N) / B` per input sample (two `N`-point FFTs and the spectral product per `B` outputs; the implementer checks the formula against the count and logs it), a row in `docs/pipeline.md` and `docs/timing.md`. `sync_detect` (backlog) is this block plus a peak search and a threshold.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

## Tests

Seeds 700 … 709.

1. Golden cases of §4 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m50.rs`).
2. Unit tests (`sdrk-core/tests/kernels_m50.rs`): `new` rejects `L = 0`, `N < 2L`, a non-power-of-two `N`, an out-of-range `N`, wrong storage lengths; an impulse input reproduces the taps (shifted by `L − 1`); a single-tap filter is a scaled delay; `block_size`; `reset`; the direct-convolution equality (`Fallback` within SNR of `Reference`) at several `L`, `N`; a length shorter than `B` produces nothing.
3. Correlation use: a known ±1 code of length 127 placed at a known offset in noise; the conj-reversed taps give the peak at that offset, with the peak-to-RMS ratio near `sqrt(L)` (seed 704).
4. Block-size invariance: 20 chunk plans (seed 705), the output within the stored SNR across chunkings (not `==`, because `fft` is tolerance).
5. Extremes: all-zero input (zero output), a full-scale impulse, NaN/∞ in one input sample (the block of that transform becomes NaN, the next clean block recovers — documented), the largest `N` (65536) with `L = 32768`.
6. Suite cases `xcorr_fft_c32` and `extremes_m50`.
7. **Block and kind** (`sdrk-blocks/tests/correlate_blocks.rs`, `sdrk-pipeline/tests/correlate_kinds.rs`; seeds 706, 707): the block equals the kernel with the conj-reversed PRBS taps on whole buffers and within the stored SNR in 20 chunk plans; a PRBS7 code with 2 samples per chip placed at a known offset in noise peaks at that offset plus `L − 1`; the automatic FFT size for `L` = 31, 127, 1023, 32767; the description rules of M09, exact storage, struct at most 512 bytes, validation errors at their parameter; a pipeline flow equals the block; `cargo xtask schema --check` and the WASM kind list include `correlate`.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entries of §5 and `correlate_block`); `cargo measure` (the same entries, every level with a core model).

## Deliverables checklist

- [ ] Kernel `xcorr_fft` (`C32`): `XcorrFftState`, trait, Fallback (overlap-save), Reference (direct), facade delegation, `backend_for`
- [ ] Contract row and section in `docs/kernels.md` (overlap-save, the correlation convention, the normalisation note)
- [ ] Golden cases (seeds 700–703) and the Python model
- [ ] Tests 1–7 (seeds 700–707); suite cases
- [ ] Count and cycle entries (`cargo xtask count`, `cargo measure`), `docs/performance.md` (break-even vs. time domain; GNU Radio / liquid-dsp method note)
- [ ] Block `CorrelateBlock` and kind `correlate` (§6): `docs/blocks.md` (entry and a new reserved-name row), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `correlate_block`
- [ ] CHANGELOG, decisions log (the no-tap-scaling reason, the tolerance class, the FFT-size rule)

## Status

Planned. Nothing is implemented.
