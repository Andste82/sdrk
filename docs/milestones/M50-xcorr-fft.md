# M50 — Kernel `xcorr_fft`: fast convolution / long correlator

> Plan: [PLAN.md › Phase 10](../PLAN.md#phase-10--digital-modes) · Depends on: [M07](M07-fft.md), [M43](M43-equalizer.md), [M49](M49-scramble.md) (the PRBS codes of the block) · Next: [M51](M51-spread.md)

> **Status: in progress** (steps 1 to 3 of the implementation delivered 2026-11-23: the kernel with fallback, reference and facade, golden cases, tests 1 to 6; the SIMD paths on every level and test 8; the block and kind `correlate` and test 7; see [Status](#status)) (written 2026-11-08, revised 2026-11-17 after an algorithm review (maintainer decision)). Kernel milestone of the DSSS/OFDM series (M48 to M56), one kernel per milestone. This is the "FFT-based fast convolution (overlap-save) as a block" item of the backlog, built now as a kernel because the sync-word correlator (`sync_detect`) needs it and any long matched filter can use it. The code-phase search (`code_search`, M53) does not call it: it computes its own circular correlation per Doppler bin and reuses only the idea.

> **Revision 2026-11-17.** The shared conventions §A to §D cited in this file are defined in [CONVENTIONS-M49-M56.md](CONVENTIONS-M49-M56.md). The maintainer accepted every recommendation of the algorithm review of M49 to M56; the shared definitions are in the shared conventions of M49 to M56 (§B.1 chip order, §C.3 `CQ15` backlog plan, §D SIMD rule). Changes in this milestone:
> 1. **Chunk invariance is bit-exact** (`==`) per backend and level (R6.2): the block boundaries are fixed by `B`, the history is a copy and the same FFTs run, so the bits cannot depend on the chunking. Only fallback, SIMD levels and facade against `Reference` (and against each other) use the SNR floor (§3, tests 4 and 7).
> 2. **SIMD in scope** (shared conventions §D, maintainer rule 2026-11-17): the kernel gets a path on every level in this milestone, the fallback's driver compiled at the level with the backend `fft` of that level and a vectorised spectral product with the fixed product formula, bit for bit except the NaN sign and payload (D296) (§5 "SIMD plan", test 8); the level-pinned inverse FFT entry of the backend (for example `fft::inverse_at`) is added here, since only `forward_at` exists. The D321 gate is a quality gate, not a reason to drop a path.
> 3. **`CQ15` is a concrete backlog item** with a plan (shared conventions §C.3) instead of the vague "backlog" line (§Scope, §7).
> 4. **`from_spectrum`**: a second constructor takes a read-only, precomputed spectrum `H` (for example a table in flash); `new` computes `H` with the `Fallback` FFT so the spectrum is the same on every backend (§2). The RAM and cost trade-off of `N = 2L` against `N = 4L` is documented (§6).
> 5. **Corrections:** M53 does not call this kernel (status line, §Scope; M50 is removed from M53's Depends line); the golden expected values are `numpy.convolve(x, h)[:rows]`, not `[L-1 : L-1+rows]` (§4); the correlation peak of a reference starting at input index `m` is at output `m + L − 1` (§2, §4, test 3); an impulse reproduces the taps without a shift and a one-tap filter is a scaled delay by its tap position, both within the SNR floor, not exactly (test 2); a NaN or ∞ makes every output of up to two blocks non-finite, not one (test 5); a call that cannot fill a whole block consumes nothing, the history holds only `L − 1` samples (§1); the automatic FFT size is capped at 65536 (§6); the Rust tests name their noise source (testkit `Awgn`).

## Goal

A kernel `xcorr_fft` that convolves a `C32` stream with a fixed complex filter of up to a few thousand taps by the overlap-save method, using the `fft` kernel of the backend, at a cost per output sample that grows with `log` of the filter length instead of linearly. It is the engine of a long matched filter: a preamble or spreading-code correlator whose reference is far too long for the time-domain `dot_complex` of `equalize`. The taps are given conjugated-and-reversed by the caller when a correlation (not a convolution) is wanted, so the kernel itself is a pure linear convolution.

## Why

The sample-level sync-word correlator named in the backlog, the acquisition of a DSSS code (hundreds to thousands of chips) and any long matched filter need `sum_k x[n+k] · conj(h[k])`. Done directly this is `L` complex multiply-accumulates per output sample; a 1023-chip code at a few samples per chip is thousands of taps, and the time-domain cost is prohibitive. Overlap-save computes one block of outputs with two FFTs and a spectral product, so the cost per output sample is `O(log N)`. GNU Radio (`gr::filter::kernel::fft_filter_ccc`) and liquid-dsp (`fftfilt`) both build their long filters this way; this kernel is the heap-free, bit-reproducible version for SDRk.

## Order

M50 depends on M07 (the `fft` kernel and `FftPlan`), on M43 (the `dot_complex` kernel, used only by the reference as the direct convolution to compare against) and on M49 (the PRBS polynomials `Prbs5` to `Prbs11` that the block `correlate` offers). It uses no kernel of M48 or M49 and only the chip order and chip value of the shared conventions §B.1 (not the code type of M51), so it can run in parallel with M51 to M56. M53 does not depend on M50 (revision 2026-11-17).

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

**In:** kernel `xcorr_fft` (`C32`) in `sdrk_core::kernels::xcorr` with `XcorrFftState` (constructors `new` and `from_spectrum`), Fallback (the overlap-save loop over the `fft` kernel), Reference (the direct convolution), facade delegation, contract row and section in `docs/kernels.md`; the SIMD path on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`; §5, shared conventions §D); golden cases; unit, invariance, extreme and SIMD tests; suite cases; `cargo xtask count` and `cargo measure`; CHANGELOG, decisions log; the block `CorrelateBlock` and the pipeline kind `correlate` (§6) with `docs/blocks.md`, `docs/pipeline.md`, `docs/timing.md`, the schema and the WASM kind list.

**Out:** the block and kind `sync_detect` and its peak search / threshold (later milestone; the name `sync_detect` stays reserved); `CQ15` (a concrete backlog item with the plan of §7, shared conventions §C.3: the block-floating-point FFT needs a common output scale across blocks); a frequency-offset search (that is `code_search`, M53, which computes its own circular correlation per Doppler bin and does not call this kernel).

## Specification

General rules of M19 apply.

### 1. Method (overlap-save)

The kernel computes the linear convolution `y[n] = sum_{k=0}^{L-1} h[k] · x[n-k]` of the input stream with the fixed taps `h` (length `L`), streaming and chunk-invariant. Overlap-save with an `N`-point FFT (`N` a power of two, `16 ≤ N ≤ 65536`, `N ≥ 2L`):

1. **Setup.** `H[·] = FFT_N{ h padded to N with zeros }`, stored once (caller storage, `N` complex values). `new` computes it with the `Fallback` forward FFT of `sdrk-core` (a state constructor cannot call a backend), so `H` has the same bits on every backend and level; `from_spectrum` takes a precomputed `H` read-only (§2). The block size is `B = N − L + 1` output samples per transform. The state keeps the last `L − 1` input samples as history (zero at the start and after `reset`), so output `n` is `y[n]` of the convolution with `x[n] = 0` for `n < 0`.
2. **Per transform block.** Copy the length-`N` window `w = [ history (L−1 samples) | next B input samples ]` into the scratch; `W = FFT_N{w}` (in place); `Y = W ⊙ H` (element-wise complex product, `Y[k] = W[k]·H[k]`, written by the kernel with the fixed formula `re = Wr·Hr − Wi·Hi`, `im = Wr·Hi + Wi·Hr`: the four products first, then one subtraction and one addition, no FMA, the `mul` fallback model; the kernel does not call the backend `mul`, whose SIMD form fuses (D326)); `y_block = IFFT_N{Y}` (in place); the valid outputs are `y_block[L−1 .. N−1]` (the first `L−1` are the wrapped, invalid part that overlap-save discards). Advance the history to the last `L−1` samples of `w` (a copy of input samples, exact).
3. A call runs `k = min(floor(input.len() / B), floor(output.len() / B))` blocks, consumes `k · B` input samples and writes `k · B` outputs; `Progress` counts both. `min_input` is `B`. The remainder of fewer than `B` input samples is **not** consumed (the history holds only the `L − 1` samples of the last window); the caller resubmits it with the next input. With `k = 0` the call produces nothing and consumes nothing. The block boundaries are those of the stream (every `B` input samples from the start or the last `reset`), whatever the chunking, so the same windows run through the same FFTs and the output is bit-identical for every chunking on the same backend and level (R6.2; §3).

**Correlation vs convolution.** The kernel convolves. To correlate a stream with a reference `r` (a sync word, a spreading code), the caller passes `h[k] = conj(r[L−1−k])` (reversed and conjugated); then `y[n] = sum_k conj(r[k]) · x[n−L+1+k]`, the sliding correlation, and the output sample `n` aligns the reference with the input window ending at `n`: a reference that starts at input index `m` (`x[m + k] = r[k]`) peaks at output index `m + L − 1`. The kernel does not reverse or conjugate; the block `CorrelateBlock` (§6) does (one line) and documents the alignment. This keeps the kernel a plain convolution and reuses it for pulse shaping as well.

**Scaling.** The forward `C32` FFT has `exponent` 0 and the inverse divides by `N` (M07), so `IFFT_N{FFT_N{w} ⊙ FFT_N{h}}` is the convolution with no extra factor: the kernel applies no scaling of its own. (GNU Radio folds `1/N` into its taps and liquid-dsp scales by `1/(2n)` because their IFFTs are unnormalised; ours is normalised, so no tap scaling is needed. Noted in the decisions log as the reason the taps are not pre-scaled.)

### 2. State and trait

```rust
pub struct XcorrFftState<T, S, Sh, Sc> { /* H: S (N complex, read-only after construction), history: Sh (L-1), scratch: Sc (N), taps_len: usize, fft: FftPlan<..> */ }
impl XcorrFftState<..> {
    /// `taps`: the L filter taps (already reversed/conjugated for a correlation). `fft` is a
    /// plan of size N, a power of two, 16..=65536, with N >= 2*L. Fills H from the taps with
    /// the `Fallback` forward FFT (`S: AsRef<[T]> + AsMut<[T]>`).
    pub fn new(taps: &[T], h_store: S, history: Sh, scratch: Sc, fft: FftPlan<..>) -> Result<Self>;
    /// `spectrum`: a precomputed H of length N (`S: AsRef<[T]>`, e.g. a `static` table in
    /// flash), the spectrum that `xcorr_spectrum` (or `new`) gives for the same taps and N.
    /// `taps_len` is L. The spectrum is not checked against any taps (it cannot be).
    pub fn from_spectrum(spectrum: S, taps_len: usize, history: Sh, scratch: Sc, fft: FftPlan<..>) -> Result<Self>;
    pub fn taps_len(&self) -> usize;
    pub fn block_size(&self) -> usize;  // B = N - L + 1
    pub fn spectrum(&self) -> &[T];     // H
    pub fn reset(&mut self);            // history to zero
}
/// Writes H = FFT_N{taps padded to N} with the `Fallback` forward FFT, exactly as `new` does
/// (for build scripts and offline tools that generate a `from_spectrum` table).
pub fn xcorr_spectrum<T>(taps: &[T], spectrum: &mut [T], fft: &FftPlan<..>) -> Result<()>;
pub trait XcorrFft<T> {
    fn xcorr_fft(&self, state: &mut XcorrFftState<..>, input: &[T], output: &mut [T]) -> Result<Progress>;
}
```

- `L = 0`, `L > N/2` (so `N < 2L`), an `N` outside the FFT range or not a power of two, or storage of the wrong length give `InvalidParameter` / `LengthMismatch` in `new` and `from_spectrum` (the same rules; `from_spectrum` checks `spectrum.len() == N`).
- `from_spectrum` with the spectrum that `new` stored for the same taps and `N` gives a state whose output equals that of `new` bit for bit (test 2).
- Timing: ratio `1/1`, delay `L − 1` input samples, the correlation alignment of §1 (a reference that starts at input index `m` peaks at output index `m + L − 1`; the block documents it). `min_input` `B`.
- Storage: `H` (`N` complex), the history (`L − 1`), the scratch (`N`, the FFT works in place on the window copy; the real-FFT-style scratch of M07 is not needed because this is a complex FFT), and the FFT plan's twiddles. `storage_bytes` exact. For `N ≤ 4096` the twiddles can come from `FftPlan::from_static_4096` (feature `static-twiddles`, flash), and with `from_spectrum` `H` can be a `static` too, so only the history and the scratch need RAM (§6).

### 3. Contract

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `xcorr_fft` | `kernels::xcorr::XcorrFft` | `C32` | tolerance | SNR ≥ 100 dB vs. the f64 direct convolution of the reference (the `fft` kernel is tolerance; the error is that of one forward FFT, a spectral product and one inverse FFT; a NumPy model of the fallback arithmetic measured 131.9 to 134.3 dB for `L`/`N` from 65/256 to 4095/16384 in the review, so the floor has about 30 dB of margin); `H` is the `Fallback` FFT of the taps on every backend; the spectral product is the fixed formula of §1 (no FMA) and the history copy is exact on every level, so the fallback and every SIMD level differ only through the `fft` of their level (shared conventions §D.3): fallback, facade and every SIMD level each reach the 100 dB floor against `Reference` and agree with each other within it; **chunk invariance is bit-exact** (`==`) per backend and level (identical block boundaries and the same FFTs for every chunking, R6.2, as `channelize`) |

**Note on bit-exactness.** Because `fft` is a tolerance kernel and the backend FFT may use another twiddle source than the fallback (M07), `xcorr_fft` is tolerance, not bit-exact across implementations: `Fallback` against `Reference`, the SIMD levels against `Fallback` and `Reference`, and the facade against both compare within the stored SNR floor. Within one backend and level everything is deterministic: the chunk-invariance tests (tests 4 and 7) and the state after a call (the history, a copy of input samples) compare with `==`. A tolerance check there would hide an off-by-one in the history advance or a shifted block phase after a resubmission. `Reference` computes `y[n] = sum_k h[k]·x[n−k]` directly in f64 (reusing the idea of `dot_complex` but in f64, O(N·L), size-limited like the FFT reference: the reference processes at most a few thousand output samples per case).

### 4. Golden vectors

New file `tests/golden/sdrk_golden/cases/xcorr.py`. Each input is four blocks long (`n_in = 4 B`, `B = N − L + 1`), so `rows = 4 B` outputs. Expected values: the f64 direct convolution of the stored taps and input with zero history, `numpy.convolve(x, h)[:rows]` (output `n` is `y[n] = sum_k h[k] x[n − k]` with `x[n] = 0` for `n < 0`, §1), independent of the FFT path. (The first draft sliced `[L-1 : L-1+rows]`, which is the output shifted by `L − 1` and does not match the kernel; corrected 2026-11-17, checked with a NumPy overlap-save model: `[:rows]` agrees to 3·10⁻¹⁴, the shifted slice does not.)

| Case | Seed | `L` | `N` | Input | Note |
| --- | --- | --- | --- | --- | --- |
| `xcorr_fft/code_127_n512` | 700 | 127 | 512 | a 127-chip ±1 code (chip value `1 − 2b` of `prbs7`, shared conventions §B.1) embedded at offset `m` in noise, the taps = conj-reversed code | correlation peak at output `m + L − 1` |
| `xcorr_fft/code_1023_n4096` | 701 | 1023 | 4096 | a 1023-chip ±1 code (`prbs10`, chip value `1 − 2b`) in noise at −10 dB chip SNR | the processing gain is 10·log10(1023) ≈ 30.1 dB, so the peak power stands about 20 dB above the mean output noise power (−10 dB chip SNR plus the gain) |
| `xcorr_fft/rrc_65_n256` | 702 | 65 | 256 | a QPSK signal, the taps = an RRC pulse | a matched filter (real taps as complex) |
| `xcorr_fft/rand_300_n1024` | 703 | 300 | 1024 | complex Gaussian input and complex Gaussian taps | general convolution, no structure |

- Tolerances in the JSON: `tolerance` with `snr_db_min` 100 (against the f64 convolution), `skip` 0 (the zero history makes the first `L − 1` outputs the exact convolution with leading zeros, so they are checked too).
- The peak-standout cases (700, 701) also store the expected peak index and the ratio of the peak magnitude to the median magnitude as information (the block's job later, but a cheap sanity check here).
- The Rust golden tests run each case in one piece and in 20 chunk plans (seed 705) and compare the chunked runs with the whole run with `==` (§3).

### 5. Cost, SIMD plan and reference rows

**Entries.** `cargo xtask count` and `cargo measure` entries `xcorr_fft_n512` and `xcorr_fft_n4096` (per output sample, every level, dominated by the two FFTs over `B` outputs), each with its `<k>_fallback` (the D321 floor). Compare in `docs/performance.md` against the time-domain cost `L` complex MACs per output (the break-even `L` where overlap-save wins is recorded from the measurement).

**Estimated work (to be confirmed by `cargo measure`).** In real flops per output, `(10 N log2 N + 6 N) / B` (two radix-2 FFTs of `5 N log2 N` each and the spectral product of `6 N`) against `8 L` for the direct complex FIR: about 127 against 1016 for `L` 127, `N` 512, and about 168 against 8184 for `L` 1023, `N` 4096 (computed from the formula, not measured). The window copy adds `N` complex loads and stores per block.

**SIMD plan (`sdrk-backend-simd`).** SIMD follows the shared conventions §D (maintainer rule 2026-11-17): every lane-parallel part of the kernel gets a path on every level in this milestone, bit-identical to the fallback through the fixed element-wise formula or reduction order of the contract; D321 is a quality gate (a slower path is improved, not removed; delegation to the fallback only after several measured failed attempts, with an open item); cycles per sample with `cargo measure` against the fallback and the fair reference (D32), unfair rows marked. For `xcorr_fft` this means (shared conventions §D.3, the `channelize` pattern of D918):

- The SIMD path is the fallback's overlap-save driver compiled at the dispatched level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`), with the backend `C32` FFT of the same level for both transforms: `fft::forward_at` (exists, used by `channelize`) for the forward transform and a level-pinned inverse entry (for example `fft::inverse_at`, with the `1/N` scaling of the fallback inverse) that this milestone adds to `sdrk-backend-simd/src/fft/c32.rs`, since only the forward entry exists today.
- The spectral product `Y = W ⊙ H` is vectorised at every level with the formula of §1 (four products, one subtraction, one addition, **no FMA** even where the level has it), so its bits equal the scalar formula on every level for every non-NaN result. The sign and payload of a NaN result (for example `∞ · 0` from a special-value input) are not part of the contract (D296); NaN lanes are compared by class. It is one helper in the backend, tested on its own against the scalar formula.
- The window copy and the history advance are plain vector copies (exact).
- No part of the kernel is serial: the FFT stages, the product and the copies are all lane-parallel. Only the FFT of the level makes the output differ from the fallback; this is the tolerance exception of §D.3 stated in the contract row (bit identity per backend and level including chunk invariance, the SNR floor against `Reference`).
- `backend_for("xcorr_fft", ..)` reports the SIMD level. A level whose path is slower than its fallback on the reference or the counter model is improved (for example by fusing the product into the last pass of the forward FFT or the first pass of the inverse, if the FFT of the level allows it with the same product formula); only after several measured failed attempts (numbers in `docs/decisions-log.md`) may it dispatch to the fallback, with an open item in `docs/open-items.md` and a `cycle_waiver` naming it.
- Planned backends (notes only): `rvv` and `arm-dsp` take the same driver with their `fft` once it exists.

**Reference rows (D32).** liquid-dsp `fftfilt_cccf` is the **reference** (the same work: streaming complex FFT fast convolution with complex taps in f32; liquid uses overlap-add, ours overlap-save, which does not change the work per output), transcribed as `benches/refs/liquid-fftfilt.c` in the two builds of `benches/refs/liquid-channelizer.c`: with liquid's own FFT (portable C) as the reference of `<k>_fallback`, and with FFTW (`SDRK_LIQUID_FFTW 1`, FFTW choosing the code of the level) as the reference of the SIMD levels where FFTW has code for that level (`sse2` and above on x86-64, `neon`); at the same `L` and the nearest block length. A gap above 10 % is an open item. GNU Radio `fft_filter_ccc` (FFTW plus VOLK `volk_32fc_x2_multiply_32fc`) is the same method and the same work as the FFTW build of liquid and is not added as a separate row (it would duplicate that row); noted in `docs/performance.md`.

**Measurement (D32, D33, D68).** The entries above and the block entry of §6 are measured in cycles with `cargo measure` on every level with a core model (reference and counter model) and counted with `cargo xtask count` on all four platforms (`--merge`); cycles are the speed measure, instruction counts stand only where no core model exists (`simd128`). A library reference enters only where it does the same work (D32); otherwise the rows are information. The D321 gate applies to every SIMD level.

### 6. Block and kind `correlate`

`CorrelateBlock` (`C32` → `C32`) around the kernel `xcorr_fft`: the sliding correlation of the input with a reference `r` of length `L`, `y[n] = sum_k conj(r[k]) · x[n − L + 1 + k]`. The block builds the taps `h[k] = conj(r[L − 1 − k])` once (§1) and chooses the FFT size. The reference is a PRBS code (`sdrk_proto::prbs`, chip `b` → `1 − 2b`, imaginary part 0) with every chip repeated `samples_per_chip` times; `CorrelateBlock::with_reference` takes any `C32` reference from the caller (Rust API only, the pipeline has no list parameters).

| Kind | Structs | In → Out | Parameters (unit, mutability, default; range) | Kernel and notes |
| --- | --- | --- | --- | --- |
| `correlate` | `CorrelateBlock` | C32 → C32 | `code` (Choice, build, `prbs7`; `prbs5` … `prbs15`), `seed` (I64, build, 1; 1 … 32767, the same name, range and meaning as the PRBS start value of `scramble`, whose `key_seed` the same maintainer decision of 2026-11-23 renames to `seed`; amended from `code_seed`), `samples_per_chip` (I64, build, 1; 1 … 16), `fft_size` (I64, build, 0 = automatic; else a power of two, 16 … 65536, at least `2 L`) | `xcorr_fft`; automatic size: the smallest power of two ≥ `4 L`, at least 16 and at most 65536 (so `L` from 16385 to 32768 gets 65536); `L` above 32768 or a manual size below `2 L` is `invalid_value` at `params.fft_size` with the required size in the message |

- Chip order and value follow the shared conventions §B.1: chip `i` is the `i`-th `Prbs::next_bit` after `Prbs::new(poly, seed)`, chip value `1 − 2b`. The block keeps no code after it has built the taps, so it uses neither the `Code` type nor anything of M51.
- `CorrelateConfig`. Storage: `H` (`N`), the history (`L − 1`), the scratch (`N`) and the twiddles of the plan; exact. Timing ratio 1, delay `L − 1` (from the kernel state), `min_input` `B = N − L + 1`; the output is tolerance class against the f64 model, and the chunk tests compare with `==` (bit-exact chunk invariance, as `channelize`).
- **Memory and cost trade-off** (documented in the block's rustdoc and `docs/blocks.md`; figures computed from the storage sizes and the cost formula below, to be confirmed by `cargo measure`): a 1023-chip code at 2 samples per chip (`L` 2046) takes about 176 KiB with the automatic `N` 8192 (`H` 64 KiB, scratch 64 KiB, history 16 KiB, twiddles 32 KiB) and about 96 KiB with `fft_size` 4096 = the smallest `N ≥ 2 L` (twiddles 16 KiB, in flash with `static-twiddles`), at about 1.4 times the cost per sample (about 128 against 92 MACs per input sample by the formula below). The cost-minimal power of two by the formula is larger than `4 L` for most `L` (32768 for `L` 2046) and saves at most about 10 % over `4 L` for `L` up to 4095, at many times the memory, so the automatic rule stays `4 L`. In the Rust API `XcorrFftState::from_spectrum` puts `H` in flash as well.
- Pipeline kind `correlate` (`c32` only), cost `(5 N log2(N) + 4 N) / B` per input sample in multiply-accumulates (the unit of `docs/pipeline.md`: two `N`-point FFTs of `2.5 N log2 N` each, as `spectrum` counts one, and the spectral product of `4 N` per `B` outputs; the implementer checks the formula against the count and logs it), a row in `docs/pipeline.md` and `docs/timing.md`. `sync_detect` (backlog) is this block plus a peak search and a threshold.
- Common rules of M29 apply: the block calls the kernel through the facade `Sdrk`, once per `process`, limited by the output room; `timing()` takes ratio and delay from the kernel state; `storage_bytes` is exact; the struct is at most 512 bytes; the description has a summary and details, and every parameter its unit, mutability, default and range.

### 7. `CQ15` (backlog item, not built in this milestone)

On Cortex-M0 and Cortex-M33 parts without an FPU a `C32` overlap-save runs in soft-float, so these targets have no usable long correlator until a `CQ15` variant exists (Cortex-M4F is served by `C32`). The backlog entry in `docs/milestones/BACKLOG.md` (entered with this revision) replaces the old "`CQ15` later" line with the plan of the shared conventions §C.3:
- the M07 block-floating-point `CQ15` FFT forward and inverse;
- the tap spectrum `H` stored as `CQ15` with its own exponent;
- per block, the exponents of the forward FFT, of `H` and of the inverse added up and applied to a caller-chosen fixed output scale with one saturating rounding shift (`round_shift` of R7, one `sat16`; shared conventions §C), so that consecutive blocks share one scale;
- the spectral product as the bit-exact `mul` `CQ15` formula;
- tolerance class, the SNR floor to be fixed by measurement (the review's estimate 40 to 50 dB, to be confirmed).

`CQ15` growth follows the shared conventions §C: sums are exact in `i64` and leave through one rounding shift `round_shift(acc, f + shift)` with one saturation (default `shift = ceil(log2 sf)`, introduced in M52); transforms shift by 0 or 1 per stage with R7 rounding and one saturation, the exponent returned as `FftScale` (the M07 FFT rule, introduced for `fwht` in M56). For short binary codes (`L` up to about 64) a direct add/subtract correlator (`despread`-style sums with the exact signs of §A, M52) may be cheaper than any FFT path; the backlog entry notes it.

## Tests

Seeds 700 … 709.

1. Golden cases of §4 through the facade, `Fallback` and `Reference` (`sdrk-testkit/tests/golden_m50.rs`), in one piece and in 20 chunk plans (seed 705) through the facade and `Fallback`, the chunked runs equal to the whole run bit for bit.
2. Unit tests (`sdrk-core/tests/kernels_m50.rs`): `new` and `from_spectrum` reject `L = 0`, `N < 2L`, a non-power-of-two `N`, an out-of-range `N`, wrong storage lengths; an impulse at input index `m` reproduces the taps at outputs `m … m + L − 1` within the 100 dB SNR floor (no shift; the kernel is a plain convolution; through the FFT path the result is not exact, so no `==` check); taps with a single non-zero tap `h[d] = c` give `c · x[n − d]`, a scaled delay by `d`, within the same SNR floor; `block_size`; `reset` (the output after `reset` equals that of a fresh state bit for bit); `xcorr_spectrum` equals the `H` of `new` bit for bit, and `from_spectrum` with that `H` gives the output of `new` bit for bit; the direct-convolution equality (`Fallback` within SNR of `Reference`) at several `L`, `N`; an input or output shorter than `B` produces nothing and consumes nothing; a call with room for `k` blocks consumes exactly `k · B` samples.
3. Correlation use: a known ±1 code of length 127 starting at a known input index `m` in noise (testkit `Awgn`, statistical only); the conj-reversed taps give the peak at output `m + L − 1`, with the peak-to-RMS ratio near `sqrt(L)` (seed 704).
4. Block-size invariance: 20 chunk plans (seed 705), the output and the history after each call **equal bit for bit** (`==`) to the whole run on the same backend (R6.2; the block boundaries do not depend on the chunking, §1); output room of 0 to 3 blocks per call included.
5. Extremes: all-zero input (exact zero output), a full-scale impulse, NaN/∞ in one input sample: every output of each block whose window holds the sample is non-finite (the FFT spreads it over the whole window). That is the block that consumes it and, if it is among the last `L − 1` samples of that block, also the next one through the history, so at most two blocks (`L − 1 < B` because `N ≥ 2L`). Every later block equals the output of the same input without the bad sample bit for bit. The largest `N` (65536) with `L = 32768`.
6. Suite cases `xcorr_fft_c32` and `extremes_m50`.
7. **Block and kind** (`sdrk-blocks/tests/correlate_blocks.rs`, `sdrk-pipeline/tests/correlate_kinds.rs`; seeds 706, 707): the block equals the kernel with the conj-reversed PRBS taps (chip order and value of the shared conventions §B.1) bit for bit on whole buffers, and bit for bit (`==`) against its own whole run in 20 chunk plans; a PRBS7 code with 2 samples per chip starting at a known input index in noise (testkit `Awgn`) peaks at that index plus `L − 1`; the automatic FFT size for `L` = 31, 127, 1023, 32767 (128, 512, 4096, 65536); the description rules of M09, exact storage, struct at most 512 bytes, validation errors at their parameter; a pipeline flow equals the block; `cargo xtask schema --check` and the WASM kind list include `correlate`.
8. **SIMD** (`sdrk-backend-simd/tests/xcorr.rs`; seed 708; shared conventions §D.1 item 6 with the §D.3 exception): the spectral-product helper of every level equals the scalar formula of §1 bit for bit, NaN lanes by class (D296: ∞ · 0 and other NaN-producing products may differ in NaN sign and payload between levels and from the constant-folded scalar; every non-NaN lane is compared by bits, as `same_f32` in `sdrk-backend-simd/tests/math.rs`), on lengths 0 to 67 and 4096 and on special values (±0, ±∞, NaN, `f32::MAX`, subnormals); the kernel at every level reaches 100 dB against `Reference` and agrees with `Fallback` within the floor for the `L`, `N` of §4 and at `N` 16 and 65536; the history after each call equals the fallback's bit for bit; chunk invariance at every level with `==` in 20 chunk plans (seed 705); the golden cases through each level; `backend_for("xcorr_fft", ..)` reports the level.

## Acceptance

R1.2 commands; `python tests/golden/generate.py --check`; `cargo test -p sdrk-core --all-features`; `cargo test -p sdrk --all-features`; `cargo test -p sdrk-testkit --features std`; `cargo test -p sdrk-backend-simd --all-features` (test 8, on the levels of the build host; CI runs every level); `cargo test -p sdrk-blocks --all-features`; `cargo test -p sdrk-pipeline --all-features`; `cargo xtask schema --check`; `cargo xtask count --platform x86_64` (the entries of §5 with their `_fallback` entries, and `correlate_block`); `cargo measure` (the same entries, every level with a core model); `cargo xtask check cycles` (D321 for every SIMD level, a `cycle_waiver` only with an open item).

## Deliverables checklist

- [x] Kernel `xcorr_fft` (`C32`): `XcorrFftState` (`new`, `from_spectrum`), `xcorr_spectrum`, trait, Fallback (overlap-save, the fixed spectral-product formula), Reference (direct), facade delegation, `backend_for` (reports `fallback` until the SIMD step)
- [x] SIMD path on every level (`sse2`, `sse4.2`, `avx2`, `avx512`, `neon`, `simd128`; §5): the driver at the level with the backend `fft` (new level-pinned inverse entry, for example `fft::inverse_at`), the vectorised spectral product without FMA (NaN lanes by class, D296); `backend_for` reports the level
- [x] Contract row and section in `docs/kernels.md` (overlap-save, the correlation convention and alignment `m + L − 1`, the normalisation note, bit-exact chunk invariance per backend and level, the §D.3 tolerance exception, `from_spectrum`)
- [x] Golden cases (seeds 700–703, expected `numpy.convolve(x, h)[:rows]`) and the Python model
- [x] Tests 1–8 (seeds 700–708); suite cases
- [ ] Count and cycle entries (`cargo xtask count`, `cargo measure`) with their `_fallback` entries (the entries exist and are measured on every level, D2610; the reference rows and the limits are open), `docs/performance.md` (break-even vs. time domain; D321 per level); the reference rows against liquid-dsp `fftfilt_cccf` in both builds (`benches/refs/liquid-fftfilt.c`), the note on GNU Radio `fft_filter_ccc`
- [x] Block `CorrelateBlock` and kind `correlate` (§6): `docs/blocks.md` (entry with the memory and cost trade-off, and a new reserved-name row), `docs/pipeline.md`, `docs/timing.md`, schema, WASM kind list; count entry `correlate_block` (the entry exists; its limits row and the results files come with the count step)
- [ ] `docs/milestones/BACKLOG.md`: the `CQ15` item (entered with the 2026-11-17 revision) still matches §7, with the measured SNR floor once known
- [ ] CHANGELOG, decisions log (the no-tap-scaling reason, the tolerance class with bit-exact chunk invariance, `H` from the `Fallback` FFT, the FFT-size rule and its cap, the spectral product without FMA, any SIMD level delegated after measured attempts)

## Status

In progress. Revised 2026-11-17 after an algorithm review (maintainer decision); the changes are listed in the revision note at the top. Maintainer decision 2026-11-23: the PRBS start value of the block `correlate` is named `seed` (not `code_seed`), and the parameter `key_seed` of the kind `scramble` is renamed to `seed` (one name per meaning, M44.1.1; a hard rename without an alias); §6 is amended in the step that builds the block.

**Step 1 (2026-11-23, D2600 to D2607): kernel, fallback, reference, golden cases, tests 1 to 6.**
- `sdrk_core::kernels::xcorr`: `XcorrFftState<T, S, Sh, Sc, Tw>` (`new`, `from_spectrum`, getters, `reset`, `Timing` with ratio 1 and delay `L - 1`), `xcorr_spectrum`, the trait `XcorrFft`, `spectral_product`/`spectral_product_one` (the fixed formula without FMA), the driver `xcorr_fft_with` and the block `overlap_save_block` (public for the SIMD step). `Fallback` (overlap-save over the fallback FFT), `Reference` (the f64 direct convolution with the taps recovered from `H`, D2600), the facade delegation to `Fallback` (`backend_for` reports `fallback`, D2606), the contract row and section in `docs/kernels.md`.
- Golden cases `xcorr_fft/{code_127_n512, code_1023_n4096, rrc_65_n256, rand_300_n1024}` (seeds 700 to 703; the exact convolution with correctly rounded sums, D2605). Measured against them (whole runs; the facade equals the fallback): fallback 134.54, 132.56, 135.94 and 133.29 dB, reference 145.35, 143.84, 147.00 and 142.76 dB; regression guards 1 dB below the fallback.
- Tests 1 to 6: `sdrk-testkit/tests/golden_m50.rs` (test 1, chunk plans seed 705; test 3, seed 704, peak-to-RMS 10.24 to 11.97 against `sqrt(127)` = 11.27), `sdrk-core/tests/kernels_m50.rs` (tests 2, 4 and 5; signals seed 709, D2607), `sdrk/tests/facade.rs`, the suite cases `xcorr_fft_c32` (quick) and `extremes_m50` (full) with `sdrk-testkit/tests/suite_m50.rs`.

**Step 2 (2026-11-23, D2608 to D2611): the SIMD paths, test 8.**
- `sdrk-backend-simd`: `fft::inverse_at` (the level-pinned inverse, `1 / N`, beside `forward_at`), `xcorr.rs` (the fallback's driver at the dispatched level with the FFT of the level, the spectral product on whole vectors without FMA, public as `xcorr::spectral_product`, vector copies); the facade delegates `xcorr_fft` to the SIMD backend and `backend_for` reports it (D2608). The driver and the fallback block copy with element loops instead of `memcpy` (D2609, the values unchanged).
- Golden cases at AVX2: 135.32, 133.28, 136.62 and 133.99 dB (fallback 134.54, 132.56, 135.94, 133.29).
- Test 8 in `sdrk-backend-simd/tests/xcorr.rs` (seed 708, D2611), run natively at AVX2, at AVX-512 under SDE and at NEON under QEMU (golden cases 134.85, 132.84, 135.35, 133.60 dB at AVX-512 and 134.79, 132.73, 135.82, 133.39 dB at NEON).
- Count and measure entries `xcorr_fft_n512`, `xcorr_fft_n4096` and their `_fallback` entries (D2610). Cycles per sample, whole-loop, reference / counter model, SIMD against the fallback: `xcorr_fft_n512` `sse2` 21.86 / 20.46 against 84.62 / 84.45, `sse4.2` 21.89 / 20.44 against 83.82 / 82.20, `avx2` 7.62 / 7.16 against 55.11 / 51.35, `avx512` 7.63 / 5.52 against 51.51 / 45.91, `neon` 44.25 / 30.68 against 88.92 / 86.67; `xcorr_fft_n4096` `sse2` 25.45 / 24.00 against 104.47 / 101.78, `sse4.2` 25.47 / 24.03 against 104.16 / 102.27, `avx2` 8.58 / 8.01 against 68.67 / 63.49, `avx512` 9.11 / 6.21 against 64.03 / 56.94, `neon` 52.10 / 35.71 against 112.23 / 114.36; `simd128` instructions per sample 186.81 against 517.52 and 217.07 against 693.01. D321 holds on every level.

**Step 3 (2026-11-23, D2612 to D2617): the block and kind `correlate`, test 7.**
- `CorrelateBlock` (`new` from `CorrelateConfig { code, seed, samples_per_chip, fft_size }`, `with_reference` for any `C32` reference), `correlate_fft_size` (automatic `4 L`, capped at 65536; the required size in the error), exact storage, the memory and cost trade-off in the rustdoc and `docs/blocks.md`; the pipeline kind `correlate` (85 kinds, cost `(5 N log2 N + 4 N) / B`, checked against the count, D2614), the schema, the WebAssembly kernel list, `docs/pipeline.md`, `docs/timing.md`; the parameter is `seed` (maintainer decision 2026-11-23, §6 amended). A pipeline stream holds at least the smallest output of its writer (D2617), so a block size above `max_chunk` works. The count entry `correlate_block`.
- Test 7: `sdrk-blocks/tests/correlate_blocks.rs` (seed 706) and `sdrk-pipeline/tests/correlate_kinds.rs` (seed 707).

Open: the reference rows against liquid-dsp `fftfilt_cccf`, the limits (also of `correlate_block`) and `docs/performance.md` (§5), the `CQ15` backlog check (§7).
