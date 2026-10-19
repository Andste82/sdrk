# Backends

A backend implements kernel traits of `sdrk-core` for the variants it accelerates. The facade `sdrk` picks, per kernel and sample type, the implementation that runs ([kernels.md](kernels.md)); `sdrk::backend_for(kernel, sample_type)` reports its name (`"simd"`, `"fallback"` or `"chained"`). Backends depend only on `sdrk-core` and `sdrk-device`, never on each other.

Today there is one backend, `sdrk-backend-simd` (layer `crates/backends`, `no_std`, no heap). The embedded backends come in later milestones.

## `sdrk-backend-simd`

`pub struct Simd;` implements the kernel traits for the variants of the table below. Everything else (the other kernels, formats and filter shapes the vector code does not cover) is delegated by the backend itself to `sdrk_core::Fallback`, so a call that the vector path cannot take returns exactly the result and the errors of the fallback.

The vector code is written once against the `Simd` trait of [`fearless_simd`](https://crates.io/crates/fearless_simd) 1.1.0 (pinned; the crate is safe Rust, so `sdrk-backend-simd` has no `unsafe` code and no module that needs the unsafe policy of the milestone). Its levels:

| Target | Levels |
| --- | --- |
| x86-64 | `sse2` (baseline), `avx2` with FMA (`x86-64-v3`), `avx512` (Ice Lake class: `avx512f`, `bw`, `dq`, `vl`, ...) |
| AArch64 | `neon` |
| wasm32 | `simd128` (compile-time only); with the cargo feature `relaxed-simd` and the target feature `+relaxed-simd` also relaxed SIMD, see below |
| anything else (Cortex-M, RISC-V, ...) | none; the facade uses `Fallback` and does not build the backend |

### Dispatch

- **Without the feature `std`** the level is fixed at compile time by the target features: `-C target-cpu=x86-64-v3` gives AVX2, `-C target-cpu=sapphirerapids` AVX-512, the default build SSE2. No detection code runs.
- **With the feature `std`** (enabled by the facade's `std`, which `sdrk-pipeline`, the examples and the testkit's `std` feature turn on) every kernel entry asks `fearless_simd::Level::new()` for the level of the running CPU. The answer is cached by the standard library inside `fearless_simd`, so it is not global state of SDRk; the per-call cost is one load and one branch. The instruction counts show it: the levels `avx2` (fixed at compile time) and `avx2-runtime` (detected at run time) differ by at most 0.15 instructions per sample for 4096-sample calls, except the two FFTs (1.2 to 1.8 more, because the same code is inlined differently; [performance.md](performance.md)).
- `--cfg sdrk_fallback_only` (set through `RUSTFLAGS`, **not** a cargo feature, because `cargo test --all-features` would switch a feature on and silently disable the backend) makes the facade use the fallback everywhere. `sdrk::SIMD_BACKEND` tells whether the accelerated kernels run on the backend in this build.
- The facade feature `simd` (default) can be switched off to drop the backend.

`sdrk_backend_simd::level_name()` returns the name of the level the kernels run at.

### Kernels and their contract

"Bit-exact" means the output equals the output of `Fallback` bit for bit; "tolerance" means the minimum SNR of `docs/kernels.md` against the f64 reference. Block-size invariance holds for all of them: an output sample depends only on its window and the state, whatever the block boundaries (the tests compare random chunk plans bit for bit).

| Kernel | Types | Contract of the backend | How |
| --- | --- | --- | --- |
| `convert` | `C32`, `CQ15` | **bit-exact** | `U8`, `I8`, little-endian `I16` (full container, power-of-two scale, interleaved, no I/Q swap): widen, convert, multiply by the exact reciprocal of the power-of-two scale. Other formats and invalid calls go to the fallback. |
| `convert_out` | `C32`, `CQ15` | **bit-exact** | `I8`, `I16`: `floor(v + 0.5)` as the nearest integer with the ties rounded up (`(v + 1.5 * 2^23) - 1.5 * 2^23` rounds to nearest even, one compare repairs the ties), NaN as 0, saturating narrow. `U8`: the sum `x * 128 + 127.5 + 0.5` rounds differently in `f32` and `f64`, so it is computed in `f64` lanes like the specification. `CQ15`: integer shifts. |
| `rotate` | `C32` | tolerance (120 dB; reaches about 140 dB) | the oscillator is evaluated with two short polynomials (`sin(pi/2 x)`, degree 9, `cos(pi/2 x)`, degree 10) on the fraction of a quarter turn, the quadrant by the top two bits; FMA where the level has it. The phase stays the `u32` accumulator. |
| | `CQ15` | **bit-exact** | the table lookup and the integer interpolation of the fallback, in `i32` lanes; per-lane table loads. |
| `fir` | `C32` | tolerance (110 dB; about 140 dB) | see below |
| | `CQ15` | **bit-exact** | exact `i16 x i16` products in `i32`, summed in `i64` lanes; filters of up to 256 taps (longer ones run on the fallback) |
| `fir_decimate` | `C32`, `CQ15` | as `fir` | outputs at the decimation positions only |
| `fir_interpolate`, `resample` | `C32`, `CQ15` | as `fir` | polyphase; the prepared taps of all branches must fit 384 samples, else the fallback runs |
| `mag_sq` | `C32` | **bit-exact** (the two squares and their sum round exactly like the fallback; there is no FMA), except in the relaxed WASM build | deinterleave, square, add |
| | `CQ15` | **bit-exact** | `u32` lanes (the sum of two squares is at most 2^31) |
| `quad_demod` | `C32` | tolerance (100 dB; about 144 dB) | vectorised `atan2`: octant reduction to `[0, 1]`, second reduction to `[0, tan(pi/8)]`, the Cephes `atanf` polynomial, the octant folded into one addition `K + S v`. On levels with 4 `f32` lanes (SSE2, NEON, SIMD128) the `atan2` runs in `f64` lanes with a degree-6 polynomial of maximum error 8e-12 (there the vector code is far cheaper than the libm-quality references, and the accuracy matches them). |
| | `CQ15` | tolerance (70 dB; at most 2 LSB from the reference) | products in `i32`, the argument in `f32` |
| `fm_mod` | `C32`, `CQ15` | **bit-exact** | the phase steps `floor(x * k + 0.5)` in `f64` lanes (magic-number conversion to the wrapped `u32`, scalar fallback for steps beyond 2^51 and NaN), a wrapping prefix sum for the phase, then the table lookup and interpolation of the fallback with the same order of `f32` operations (no FMA) |
| `fft` | `C32` | tolerance (110 dB; about 137 dB) | decimation in time, in place; see below |
| | `CQ15` | **bit-exact**, the block-floating-point exponent too | the radix-2 stages of the fallback in `i32` lanes (the twiddle products and the butterfly sums of a plan made by `FftPlan::new` stay below 2^31) |

Not accelerated (they stay with the fallback): `scale`, `window`, `power_db`, `deemphasis`, `dc_block`, `dot`, `dot_real`, `convert_real`, `convert_out_real`, `resample_real`, `fir_interpolate_real`, `rfft`, `irfft`. The fused kernels `ddc`, `duc` and `fm_demod` are chains of the facade's own kernels, so they use the accelerated parts.

**Dot products (`fir` family).** Every output is the dot product of the taps with a window of the input. The window may reach back into the history, so the input is the virtual stream `history ++ input`. The taps are prepared once per call, reversed and with every tap written twice (real and imaginary lane), so the data is read in memory order and every output reuses the same tap vectors. A chunk is half a vector of complex samples. Chunk `c` goes to accumulator `c % 4` (four short chains are more accurate than one long chain, and the dependency chains are shorter), the partial last chunk to the one it would belong to, and the four accumulators are added in a fixed order; a window that reaches into the history is assembled chunk by chunk with the same boundaries, so an output is bit-identical whichever way the input was split. `fir` and `fir_decimate` prepare the taps in blocks of 256 taps (one block for the usual filters; a longer filter adds the partial dot products of the blocks).

**FFT.** `C32`: bit-reversal permutation by `V x V` tiles (`V` complex samples per vector; the elements with a fixed middle part of the index form a tile of `V` vector loads that goes, transposed and reversed in both directions by a `zip` network, to the tile of the reversed middle part: about 8 instructions per sample instead of 30 for the scalar loop); conversion to blocks of `L` real parts followed by `L` imaginary parts and the first `log2 L` radix-2 stages inside the block with lane permutations; then two radix-2 stages at a time as one four-point butterfly (radix-2^2, three twiddle sets, one extra radix-2 pass first when the number of stages is odd), vectorised over `L` consecutive butterflies; the twiddles of `L` consecutive butterflies are gathered from the plan's table as 64-bit words; conversion back to interleaved samples with the inverse scale `1/N`. The spec asks for radix-4: the fused radix-2^2 pass is the radix-4 butterfly structure on the bit-reversed order (decimation in time in place), see the decisions log. Without per-stage twiddle tables and without scratch (a plan is one table of `N/2` twiddles and the call has the data only) the pass cannot reach the instruction counts of rustfft, see [performance.md](performance.md).

**FMA.** Used by the tolerance kernels on levels that have it (`rotate` `C32` and the `C32` FFT, nothing else); never in a bit-exact kernel, and the `fir` family and `quad_demod` use separate multiplications and additions on every level. Because FMA and the vector width change the roundings, the tolerance kernels give different bits on different levels (and differ from the fallback); the contract bounds the difference.

### Relaxed SIMD (wasm32)

The cargo feature `relaxed-simd` of `sdrk-backend-simd` (forwarded by the facade `sdrk` and by `sdrk-wasm`) is for builds with `RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd"`. In such a build the `C32` variants of `fir`, `fir_decimate`, `fir_interpolate` (`resample`) and `mag_sq` accumulate with the relaxed multiply-add (`f32x4.relaxed_madd`), and `rotate` and the `C32` FFT, which use `mul_add` on every level, get it from `fearless_simd` (it uses the relaxed instruction for `mul_add` whenever the target feature is on). The engine may fuse the multiplication and the addition or not, whichever is faster on its hardware: the results are **not bit-exact** and differ between hosts, but stay within the contracts of the table (the tolerance kernels), and `mag_sq` stays within a few ulp of the fallback. On every other target, and on wasm32 without the target feature, the feature changes nothing. The block-size invariance of the `fir` family holds in the relaxed build as well, because every path through the window uses the same instruction.

`cargo xtask wasm-pkg --relaxed` builds this variant as `sdrk_relaxed.wasm`; it is never selected automatically ([wasm.md](wasm.md)). The CI job `wasm-wasmtime` runs the tests of the backend, the facade and the blocks with it under wasmtime.

**Closures.** In `fearless_simd` code a closure that contains vector operations is not inlined into the `#[target_feature]` function if it is called more than once, and then runs without the vector instructions. The kernels therefore use `#[inline(always)]` functions and local macros, not closures, for vector code.

### Limits of the vector paths

- `convert`, `convert_out`: only the formats named above; `U8` output needs `f64` lanes.
- `CQ15` filters longer than 256 taps and polyphase filters whose branches do not fit 384 prepared samples run on the fallback (the result is the same, only slower).
- `fm_mod`: if a phase step does not fit 2^51 (a deviation of 10^15 Hz and more) or the input is NaN, the rest of the call runs on the fallback.
- The FFT sizes are those of the plan (16 to 65536); sizes below `V^2` complex samples (64 on AVX-512) use the scalar bit reversal.

## How to add a backend

1. A crate `crates/backends/sdrk-backend-<name>` that depends on `sdrk-core` (and `sdrk-device`), `#![no_std]`, implementing the traits for the variants it accelerates with `pub struct <Name>;`.
2. In the facade: `impl BackendName for <Name>` with the name, and the entries of `crates/facade/sdrk/src/delegations.rs` for the accelerated kernels (`=> <Name>`, guarded by `#[cfg(...)]`).
3. Cases `backend_vs_fallback_*` of the suite apply to the facade, so they exercise the backend automatically; add the platform to `cargo xtask count` and the CI matrix.
4. Unsafe code: only in a separate module per architecture with `#[allow(unsafe_code)]` locally, `// SAFETY:` on every block (enforced by `clippy::undocumented_unsafe_blocks`) and a line in the decisions log.
