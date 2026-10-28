# Performance

Performance is measured in executed instructions per sample in an emulator and compared with an established library (VOLK, FFTW, rustfft, KissFFT; CMSIS-DSP and ESP-DSP for the embedded backends to come) at equal accuracy against the f64 reference.

| Instructions per sample vs. reference library | Meaning |
| --- | --- |
| ≤ 1.00× | goal: equal or better (recorded as `goal` in the results) |
| ≤ 1.10× | release limit; CI fails above it for a `release` entry |
| ≤ 1.50× | allowed only while a kernel is under development (`dev`), never in a release |

Instruction counts are a strong proxy on in-order cores (Cortex-M, ESP32) and a weaker one on out-of-order CPUs and for memory-bound kernels. They say nothing about cache behaviour, the cost of one instruction (a gather is one instruction and many cycles) or the clock; they are the portable, reproducible measure of how much work a kernel does. Kernels that use gathers (`fm_mod` at `avx512`, and the NCO lookups of `rotate` `CQ15` there) need a wall-clock check on real AVX-512 hardware (Zen 4, Sapphire Rapids) before their counts are taken as a gain.

## Method

`cargo xtask count --platform <platform>` builds the measurement binary of `benches/kernels` for every level of the platform and runs it under an instruction counter.

- **Differential count.** `count run <kernel> <impl> <k>` does one warm-up call and then `k` calls on the same buffers. It is started twice, with `k = 0` and with `k = K`, and the instructions per sample are `(count(K) - count(0)) / (K * samples)`. Process start, allocation, set-up (the plans of rustfft, FFTW and KissFFT, the VOLK dispatcher) and the warm-up call cancel. `K = 16` calls of 4096 samples; a kernel with fewer samples per call (the FFT sizes 64 and 1024) gets proportionally more calls, so every measurement covers 65 536 samples.
- **Inputs.** PCG32 noise with the seed `1000 + index of the kernel`, 4096 samples, buffers aligned to 64 bytes for both sides. The FFTs are in place: every call first restores the input, and the count of the restore alone is subtracted from both sides, so the transforms always see the same finite data.
- **Variants (information).** The streaming kernels are also measured with buffers misaligned by one sample, and the conversions with blocks of 64 and 100 samples. These lines are reported in the results and never judged.
- **Counters.**

| Level | Build | Counter |
| --- | --- | --- |
| `sse2` | default x86-64 build | Intel SDE with the CPUID of a strict SSE2 x86-64 CPU (`-cpuid_in`, `-chip_check_cpuid 1`): VOLK, rustfft and FFTW choose the code of that level themselves, and any instruction above it stops the run |
| `sse4.2` | `-C target-cpu=x86-64-v2` | Intel SDE with the CPUID of an x86-64-v2 CPU (up to SSE4.2 and POPCNT), as for `sse2` |
| `avx2` | `-C target-cpu=x86-64-v3` | Valgrind `callgrind` (Linux; not available on Windows, so the count job runs on Linux only) |
| `avx2-runtime` | default build with the feature `std` (run-time detection) | Valgrind `callgrind`; judged against our own numbers at `avx2` (at most 1.02×, the cost of the dispatch) |
| `avx512` | `-C target-cpu=sapphirerapids` | Intel SDE `-spr` |
| `neon` | `aarch64-unknown-linux-gnu` | `qemu-aarch64` in user mode with the `insn` plugin and a reserved guest address space (`-R 0x200000000`), which fixes the memory layout: identical runs give identical counts |
| `simd128` | `wasm32-wasip1`, `+simd128` | wasmtime with fuel: the smallest fuel with which the run completes, found by bisection |

  Under SDE only the region between two SSC markers of the binary (`benches/kernels`, `region_start`/`region_stop`) is counted, which makes the counts exact; each count is still the median of up to three runs (the third run is made only when the first two differ). The CPUID definitions keep the XSAVE state to x87 and SSE: on a CPU without OSXSAVE the `cpu_features` library inside VOLK would read the SSE flags of the host from `/proc/cpuinfo` and select a machine that the emulated CPU does not have. SDE's own `-p4` does not run 64-bit code, so it cannot be used for `sse2`.
- **Reproducibility.** The measurement binary is built with `--remap-path-prefix` for the checkout (`/sdrk`) and the Cargo home (`/cargo`), passed through `CARGO_ENCODED_RUSTFLAGS`. Otherwise these paths end up in the panic locations of the binary, and their length moves its data and with it some counts. The two runs of a measurement get arguments of the same length: `k` is written with as many digits as the number of calls (`0000` and `1024`). The start-up code of a process depends on the length of its arguments and environment; with the same length, the start-ups are identical and cancel exactly. With both measures the counts no longer depend on the path of the checkout or on the environment of the caller: a checkout at another path gives identical lines on every platform.
- **Run time.** The levels of a platform are counted in parallel: first the levels judged against the references, then the run-time detection level, which is judged against our numbers at `avx2`. The C references are built once before. `x86_64` with the variants takes about 14 minutes instead of 20 on four cores.
- **Accuracy.** `count accuracy <kernel> <impl>` prints the SNR of one run against the f64 reference. An entry fails if its SNR is more than 1 dB below the SNR of its comparison, unless it carries an `accuracy_waiver`. The instruction ratio and the accuracy are judged independently.

### References

The C references are built by `benches/refs/build.sh <x86_64|aarch64|wasm32> <prefix>` from pinned release tarballs (each checked against its SHA-256) into `benches/target/refs/<platform>`; xtask runs the script before it builds a platform and copies its manifest (versions, compilers, flags) into the results. Only these builds are used, never a VOLK of the system: the distribution's libvolk is built differently (`-O2`, LTO, ORC) and gives other counts (with the earlier callgrind method, which ran VOLK's `generic` code compiled for AVX2 at `sse2`, `convert_cs8` needed 14.02 instructions per sample with the distribution's library against 1.65 with this build).

| Library | Version | License | Platforms | Use |
| --- | --- | --- | --- | --- |
| VOLK | 3.3.0 | LGPL-3.0-or-later | x86_64, aarch64 (shared, CMake Release, no ORC); wasm32 (the `generic` protokernels, static, `benches/refs/volk-wasm.c`) | the reference of the streaming kernels |
| FFTW | 3.3.10 | GPL-2.0-or-later | x86_64 (`--enable-sse2`), aarch64 (`--enable-neon`); static, single precision, `FFTW_ESTIMATE` | the `C32` FFT reference at `sse2` (out of place, its usual use; in place as information), information on `neon` |
| KissFFT | 131.1.0 | BSD-3-Clause | all (static, `FIXED_POINT=16`) | the `CQ15` FFT reference |
| Ne10 | 1f059a7 (last commit of the archived repository) | BSD-3-Clause | aarch64 (static, DSP module) | information next to KissFFT on `neon` |
| GNU Radio `agc_cc` | the loop of `agc.h` of GNU Radio main at `ec1bf15`, translated to C in `benches/refs/gr-agc.c` (about 20 lines, in the repository) | GPL-3.0-or-later | all (static, one object file, the generic flags of GNU Radio's build) | the `C32` `agc` reference; the only reference source that is part of the repository (it is bench-only, never linked into a library crate, and carries its license in its header) |
| rustfft | 6.4 (Cargo dependency of `benches/kernels`) | MIT OR Apache-2.0 | all; `wasm_simd` on wasm32 (target-specific) | the `C32` FFT reference except at `sse2` |
| KFR | 7.2.0 | GPL-2.0-or-later | x86_64 (GCC), aarch64 (`clang++ --target=aarch64-linux-gnu`: GCC rejects KFR's NEON code); header-only `kfr::iir` behind `benches/refs/kfr-biquad.cpp`, static, one library per level | the SIMD reference of `biquad` C32 and `biquad_real` f32 (D15) |
| CMSIS-DSP | 1.18.0 | Apache-2.0 | all; generic C (`__GNUC_PYTHON__`, `ARM_MATH_LOOPUNROLL`, the configuration of its own host build) behind `benches/refs/cmsis-biquad.c`, static, one library per level | the scalar reference of the four biquad fallbacks (D16); the gate of `biquad` C32 and `biquad_real` f32 at `simd128`; information next to KFR and next to the Q15 entries (D15) |

The libraries are linked only into the measurement binary of `benches/kernels`, which is built on the measuring machine and never distributed; none of them is a dependency of a library crate of SDRk. `benches/` is a Cargo workspace of its own that is not a member of the repository workspace (`exclude = ["benches"]`), so `cargo deny` does not see it. VOLK is linked dynamically (`benches/volk-sys`, hand-written `extern "C"` declarations; VOLK declares its kernels as function-pointer variables that its dispatcher fills in), FFTW, KissFFT, Ne10, GNU Radio's `agc_cc`, KFR and CMSIS-DSP statically through `benches/cref-sys`.

The biquad references are compiled once per level with the instruction set of the level (`-march=x86-64`, `x86-64-v2`, `x86-64-v3`, `sapphirerapids` for `sse2`, `sse4.2`, `avx2`, `avx512`; `armv8-a` for `neon`; `-msimd128` for `simd128`), all with `-O3 -ffp-contract=off` (no FMA on either side, D5), into `libkfrbq-<level>.a` and `libcmsisbq-<level>.a`; `benches/cref-sys/build.rs` links the pair of the level that the target features of the measurement build select, the rule of `sdrk-backend-simd`. The level `avx2-runtime` is judged against our `avx2` numbers and runs no reference. KFR's shim is compiled without exceptions and RTTI and allocates with `aligned_alloc`, so it needs no C++ run-time library; its filter loop throws nothing, so this does not change the measured code. KFR is not built for wasm32: it needs a C++ standard library for WASI (Emscripten), which the toolchain here does not have.

What each kernel is compared with:

| Kernel | Comparison |
| --- | --- |
| `convert` CS8 / CS16 to C32 | `volk_8i_s32f_convert_32f` / `volk_16i_s32f_convert_32f` on 2N values |
| `convert` CU8 to C32 | `volk_8i_s32f_convert_32f` on the same 2N bytes with the sign bit flipped (the same work apart from the offset of 0.5 / 128, which VOLK, having no unsigned 8-bit input, does not add); our fallback as information |
| `convert_out` C32 to CS16 / CS8 / CU8 | `volk_32f_s32f_convert_16i` / `volk_32f_s32f_convert_8i` / `volk_32f_s32f_x2_convert_8u` on 2N values |
| `rotate` C32 | `volk_32fc_s32fc_x2_rotator2_32fc` |
| `fir` C32, 64 taps; `fir_decimate` C32, 127 taps, M = 10 | `volk_32fc_32f_dot_prod_32fc` per output, called as GNU Radio's FIR filters call it (aligned entry point, taps pre-shifted, D6) |
| `fir_interpolate` C32, L = 2, 128 taps; `resample` C32, 3 / 2, 192 taps (64 per branch; per output sample) | the same dot product per output with the taps of the output's branch, as GNU Radio's `interp_fir_filter_ccf` and `rational_resampler_ccf` call it |
| `mag_sq` C32 | `volk_32fc_magnitude_squared_32f` |
| `mag` C32 | `volk_32fc_magnitude_32f` |
| `mag` CQ15 | `volk_16ic_magnitude_16i` (VOLK scales to float and back and rounds differently from the integer rule; the entries are `release` (they started as `dev`; every ratio is below 1.00, open item M20-3), and the accuracy rule compares the SNR of both against the unrounded magnitude) |
| `quad_demod` C32 | `volk_32fc_x2_multiply_conjugate_32fc` + `volk_32fc_s32f_atan2_32f` |
| `rotate` CQ15, `mag_sq` CQ15, `quad_demod` CQ15, `fm_mod` C32, `squelch` C32 and CQ15, `agc` CQ15, `dc_block_real` f32 and Q15 | our fallback (no established library); `squelch`, `agc` CQ15 and `dc_block_real` are not accelerated, so their entries are `info` |
| `biquad_real` f32, 2 sections | KFR 7.2.0 `kfr::iir` (`iir_state<float, 2>`: the sections in the lanes of one vector, transposed direct form II, the same algorithm class as ours) at `sse2`, `sse4.2`, `avx2`, `avx512` and `neon`; CMSIS-DSP `arm_biquad_cascade_df2T_f32` (generic C) as information there and as the gate at `simd128`, where KFR is not built (the D12 rule: the code a library would run there) (D15) |
| `biquad` C32, 2 sections | KFR as above on the re and im parts as two real channels: two calls of N real samples (the 2N-values rule of `convert`), the deinterleaving done once outside the counted region (the best case for KFR, which has no complex type); CMSIS-DSP `arm_biquad_cascade_stereo_df2T_f32` on the interleaved buffer (its two channels are our re/im layout) as information, and as the gate at `simd128` (D15) |
| `biquad` CQ15, `biquad_real` Q15, 2 sections | our fallback, as information only (D21): no library has a SIMD Q15 biquad on our levels (CMSIS-DSP has one for Helium/MVE only; Intel IPP's 16-bit IIR computes in float, another contract); CMSIS-DSP `arm_biquad_cascade_df1_q15` (generic C, `postShift = 1`) as information, for CQ15 two calls on planar re and im (D15) |
| `biquad_fallback`, `biquad_cq15_fallback`, `biquad_real_fallback`, `biquad_real_q15_fallback` (the measured side is our fallback, compiled for the level) | CMSIS-DSP 1.18.0 generic C, compiled for the same level (D16): `arm_biquad_cascade_stereo_df2T_f32` for C32, `arm_biquad_cascade_df2T_f32` for f32, `arm_biquad_cascade_df1_q15` with `postShift = 1` for Q15 and, on planar re and im, for CQ15. The coefficients are ours with `a1`, `a2` negated (CMSIS-DSP adds the feedback terms); with `postShift = 1` the raw Q15 coefficients are exactly our Q2.14 values. CMSIS-DSP's Q15 cascade truncates where ours rounds half up, and its f32 cascades do not flush the state; both are accepted as the same work. A feedback coefficient of -2.0 (raw -32768) cannot be negated in Q15; the measured Butterworth does not use it. Not measured at `avx2-runtime` |
| `agc` C32 | the loop of `agc_cc` of GNU Radio (`gr-analog/include/gnuradio/analog/agc.h`), a C copy in `benches/refs/gr-agc.c` compiled with the generic flags of GNU Radio's own build (`-O3`, no `-march`) for every level of the platform; our fallback is information. The `agc` of GNU Radio is a different algorithm (additive, no lower limit, no silence rule), so the accuracy column compares each with the f64 model of its own loop |
| `fft` C32, N = 64, 1024, 4096 | FFTW at `sse2`, rustfft (`FftPlanner`, forward, in place) elsewhere |
| `fft` CQ15, N = 1024, block float and fixed scaling (noise of 8000 LSB, and of full scale for fixed scaling) | KissFFT `FIXED_POINT=16`; Ne10 int16 as information on `neon` |
| `rfft` f32, N = 1024 | FFTW's real-input transform (`fftwf_plan_dft_r2c_1d`, out of place) at `sse2` and `sse4.2` (FFTW has no codelets above SSE2, so it is the stronger reference there too), rustfft's real-input planner (`realfft` 3.5, forward, out of place) elsewhere; the other of the two and our fallback as information (FFTW also on `neon`) |

VOLK's dispatcher is pinned per level by a `volk_config` that xtask writes (`benches/target/<level>/volk-pinned/volk/volk_config`), and the measured processes run without `VOLK_GENERIC` and without the variables that override single kernels:

- `sse2`, `sse4.2`: the defaults of VOLK's `sse2_64_mmx` and `sse4_2_64_mmx` machines, which the emulated CPUs select.
- `avx2`, `avx2-runtime`: the implementations that need no more than AVX2 and FMA, so that a runner with AVX-512 does not dispatch `rotator2`, `atan2` and `dot_prod` to AVX-512 code.
- `avx512`: VOLK's own choice, as our kernels use AVX-512 there too.
- `neon`: the default dispatch of VOLK's `neonv8` machine (D11), pinned so that a change of VOLK's ranking shows up as a missing implementation instead of a silently different reference. Every VOLK kernel also gets the cheapest combination of VOLK implementations of the machine as information (`goal: cheapest VOLK implementation`).
- `simd128`: VOLK has no wasm32 machine; its `generic` protokernels compiled for wasm32 with `simd128` are what its dispatcher would take. They gate `mag_sq`, `mag` (D12 extended in M20: the `generic` protokernel of `volk_32fc_magnitude_32f` is the code its dispatcher would take, as for `mag_sq`) and the conversions only (D12); for `rotate`, the FIR family and `quad_demod` they are scalar code and the entries are `info`.

### Where no fair reference exists

- VOLK has no SSE implementation of `rotator2` and `atan2`; at `sse2` and `sse4.2` the dispatcher takes their scalar `generic` code, so `rotate` and `quad_demod` are `info` at both levels. VOLK has no implementation of `multiply_conjugate` and `8i_s32f_convert_32f` below SSE3 and SSE4.1 either, but the `generic` code of the conversion is vectorized with SSE2 by the compiler, so `convert_cs8` and `convert_cu8` are judged against it at `sse2`.
- At `simd128`, see above.
- `info` entries are reported with their numbers and never judged; the reason is the `waiver` text of the entry.

## Limits

`benches/limits.json` names, per kernel and level, `{ "status": "release" | "dev" | "info" }`, and the caps in `_limits` (`release` 1.10, `dev` 1.50, `warn` 1.10, `goal` 1.00, `runtime` 1.02; D14). `cargo xtask count` writes `benches/results/<platform>.json` (counts, ratio, SNR of ours and of the comparison, the verdict and whether the entry reaches the goal) and fails if a `release` entry exceeds 1.10× or a `dev` entry 1.50×, or if an entry is less accurate than its comparison by more than 1 dB. A kernel and level that is not listed fails. The level `avx2-runtime` needs no entries: it is judged against our own `avx2` numbers with the cap `runtime`.

An entry may carry a `"waiver"` (ratio) or an `"accuracy_waiver"` text that names the open item in `docs/open-items.md` which tracks a limit that is not met yet. A waived exceedance is printed (`waived`) and recorded in the results file, but does not fail the command; the numbers are never hidden. A waiver that no longer covers anything, and a `dev` entry at or below 1.10×, are reported as a warning; a passing entry above `warn` (a `dev` entry between 1.10× and 1.50×) gets the verdict `warn`. `--strict` (the release gate of open item M13-4) additionally fails for every entry that is not a clean `release` entry: `dev`, `info`, a waiver, or a failure. The unit tests of the limit logic (`xtask/src/count.rs`) include a deliberately slow dummy kernel with status `release`, which makes the command fail.

## Results

Instructions per sample of our kernel divided by those of the comparison (lower is better), from the committed `benches/results/*.json`; `dev` marks an entry whose status is `dev`, `info` one that is reported but not judged, `waived` one with a waiver. All other entries are `release`, and every entry at or below 1.00 reaches the goal. Measured with VOLK 3.3.0, FFTW 3.3.10, KissFFT 131.1.0 and rustfft 6.4 as built by `benches/refs/build.sh` on the development machine (the counts are deterministic: two complete runs gave identical lines on every platform).

| Kernel | sse2 | sse4.2 | avx2 | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | 0.62 | 0.70 | 0.61 | 0.62 | 0.57 | 0.38 |
| `convert` CS16 to C32 | 0.43 | 0.56 | 0.61 | 0.62 | 0.48 | 0.43 |
| `convert` CU8 to C32 | 0.62 | 0.83 | 0.61 | 0.62 | 0.62 | 0.36 |
| `convert_out` C32 to CS16 | 0.91 | 0.91 | 0.80 | 0.70 | 0.60 | 0.20 |
| `convert_out` C32 to CS8 | 0.99 | 1.02 | 0.84 | 0.72 | 0.65 | 0.32 |
| `convert_out` C32 to CU8 | 1.35 dev | 1.05 | 1.06 | 0.69 | 0.96 | 0.52 |
| `rotate` C32 | 0.16 info | 0.18 info | 0.45 | 0.25 | 0.60 | 0.20 info |
| `rotate` CQ15 | 0.43 | 0.23 | 0.28 | 0.22 | 0.22 | 0.53 |
| `fir` C32, 64 taps | 0.49 | 0.48 | 0.13 | 0.06 | 0.36 | 0.11 info |
| `fir_decimate` C32, 127 taps, M = 10 | 0.51 | 0.50 | 0.21 | 0.18 | 0.52 | 0.14 info |
| `fir_interpolate` C32, L = 2, 128 taps | 0.51 | 0.51 | 0.20 | 0.13 | 0.40 | 0.12 info |
| `resample` C32, 3 / 2, 192 taps | 0.52 | 0.51 | 0.26 | 0.20 | 0.50 | 0.16 info |
| `mag_sq` C32 | 0.71 | 0.62 | 0.61 | 0.39 | 0.52 | 0.60 |
| `mag_sq` CQ15 | 0.11 | 0.15 | 0.18 | 0.18 | 0.31 | 0.13 |
| `quad_demod` C32 | 0.13 info | 0.14 info | 0.68 | 0.50 | 0.62 | 0.18 info |
| `quad_demod` CQ15 | 0.22 | 0.16 | 0.07 | 0.02 | 0.10 | 0.13 |
| `fm_mod` C32 | 0.27 | 0.22 | 0.13 | 0.06 | 0.27 | 0.32 |
| `squelch` C32 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `squelch` CQ15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `mag` C32 | 0.73 | 0.66 | 0.64 | 0.52 | 0.72 | 0.80 |
| `mag` CQ15 | 0.24 | 0.46 | 0.85 | 0.46 | 0.72 | 0.49 |
| `agc` C32 | 0.99 | 0.96 | 0.72 | 0.67 | 1.03 | 1.01 |
| `agc` CQ15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `dc_block_real` f32 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `dc_block_real` Q15 | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info | 1.00 info |
| `biquad` C32, 2 sections (KFR; CMSIS-DSP at `simd128`) | 0.21 | 0.23 | 0.25 | 0.18 | 0.27 | 0.36 |
| `biquad` CQ15, 2 sections (fallback) | 0.47 info | 0.43 info | 0.16 info | 0.20 info | 0.52 info | 0.35 info |
| `biquad_real` f32, 2 sections (KFR; CMSIS-DSP at `simd128`) | 0.36 | 0.37 | 0.49 | 0.33 | 0.55 | 0.69 |
| `biquad_real` Q15, 2 sections (fallback) | 0.49 info | 0.42 info | 0.39 info | 0.38 info | 0.52 info | 0.37 info |
| `biquad_fallback` C32 (CMSIS-DSP) | 0.56 | 0.56 | 0.53 | 0.53 | 0.52 | 0.58 |
| `biquad_cq15_fallback` (CMSIS-DSP) | 0.72 | 0.72 | 0.90 | 0.72 | 0.66 | 0.73 |
| `biquad_real_fallback` f32 (CMSIS-DSP) | 0.99 | 0.98 | 0.95 | 0.89 | 0.97 | 0.88 |
| `biquad_real_q15_fallback` (CMSIS-DSP) | 0.75 | 0.75 | 0.67 | 0.69 | 0.68 | 0.81 |
| `fft` C32, N = 64 | 0.80 | 0.60 | 0.99 | 0.95 | 0.58 | 0.56 |
| `fft` C32, N = 1024 | 0.98 | 0.74 | 0.91 | 0.50 | 0.59 | 0.64 |
| `fft` C32, N = 4096 | 0.93 | 0.73 | 0.86 | 0.50 | 0.56 | 0.64 |
| `fft` CQ15, N = 1024, block float | 0.12 | 0.10 | 0.04 | 0.02 | 0.09 | 0.11 |
| `fft` CQ15, N = 1024, fixed scaling | 0.11 | 0.08 | 0.04 | 0.02 | 0.07 | 0.09 |
| `fft` CQ15, N = 1024, fixed scaling, full scale | 0.12 | 0.09 | 0.04 | 0.02 | 0.08 | 0.10 |
| `rfft` f32, N = 1024 | 1.09 | 1.09 | 0.61 | 0.43 | 0.56 | 0.71 |

Instructions per sample, ours / comparison (`avx2-runtime`: ours only, judged against ours at `avx2`):

| Kernel | sse2 | sse4.2 | avx2 | avx2-runtime | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | 2.50 / 4.02 | 2.12 / 3.02 | 1.08 / 1.77 | 1.09 | 0.56 / 0.89 | 1.40 / 2.46 | 4.22 / 11.03 |
| `convert` CS16 to C32 | 2.38 / 5.52 | 2.12 / 3.77 | 1.08 / 1.77 | 1.09 | 0.55 / 0.89 | 1.46 / 3.02 | 4.34 / 10.03 |
| `convert` CU8 to C32 | 2.50 / 4.02 | 2.50 / 3.02 | 1.08 / 1.77 | 1.09 | 0.56 / 0.90 | 1.53 / 2.46 | 3.96 / 11.03 |
| `convert_out` C32 to CS16 | 3.67 / 4.02 | 3.67 / 4.02 | 1.71 / 2.14 | 1.72 | 0.72 / 1.02 | 2.03 / 3.39 | 5.67 / 29.03 |
| `convert_out` C32 to CS8 | 3.61 / 3.65 | 3.61 / 3.52 | 1.65 / 1.96 | 1.66 | 0.73 / 1.02 | 2.16 / 3.33 | 9.17 / 29.03 |
| `convert_out` C32 to CU8 | 5.42 / 4.02 | 5.30 / 5.02 | 2.34 / 2.21 | 2.35 | 1.52 / 2.21 | 3.97 / 4.14 | 15.17 / 29.03 |
| `rotate` C32 | 7.26 / 46.11 | 7.26 / 41.11 | 2.37 / 5.28 | 2.39 | 1.23 / 4.99 | 3.57 / 5.99 | 15.94 / 79.23 |
| `rotate` CQ15 | 24.53 / 57.02 | 13.27 / 57.02 | 7.53 / 26.53 | 7.54 | 2.78 / 12.52 | 10.77 / 50.01 | 45.04 / 84.29 |
| `fir` C32, 64 taps | 118.50 / 244.02 | 117.97 / 247.02 | 24.92 / 193.77 | 24.94 | 11.16 / 197.52 | 55.59 / 153.02 | 209.41 / 1980.03 |
| `fir_decimate` C32, 127 taps, M = 10 | 23.59 / 46.47 | 23.59 / 47.27 | 6.87 / 32.55 | 6.82 | 4.47 / 25.03 | 14.08 / 26.84 | 55.44 / 383.41 |
| `fir_interpolate` C32, L = 2, 128 taps | 133.77 / 262.02 | 134.57 / 265.02 | 41.88 / 211.77 | 41.70 | 28.01 / 215.52 | 68.54 / 172.51 | 245.10 / 2024.03 |
| `resample` C32, 3 / 2, 192 taps | 137.53 / 266.35 | 137.72 / 269.35 | 55.19 / 215.52 | 55.80 | 43.78 / 218.42 | 86.61 / 174.35 | 317.99 / 2026.03 |
| `mag_sq` C32 | 2.32 / 3.27 | 1.57 / 2.52 | 0.92 / 1.52 | 0.93 | 0.60 / 1.52 | 1.18 / 2.27 | 4.96 / 8.28 |
| `mag_sq` CQ15 | 1.31 / 11.51 | 1.31 / 8.51 | 0.73 / 4.08 | 0.75 | 0.36 / 1.95 | 1.02 / 3.26 | 3.02 / 23.90 |
| `quad_demod` C32 | 18.80 / 149.03 | 18.30 / 131.54 | 7.43 / 10.92 | 7.44 | 3.36 / 6.73 | 12.30 / 19.79 | 36.15 / 201.21 |
| `quad_demod` CQ15 | 38.15 / 170.85 | 26.77 / 171.16 | 11.27 / 162.72 | 11.29 | 4.03 / 163.66 | 17.78 / 173.61 | 42.91 / 332.25 |
| `fm_mod` C32 | 16.51 / 61.91 | 13.74 / 61.91 | 7.51 / 58.92 | 7.65 | 3.40 / 55.92 | 11.22 / 41.95 | 34.78 / 109.89 |
| `squelch` C32 | 27.35 / 27.35 | 26.35 / 26.35 | 21.84 / 21.84 | 27.35 | 21.80 / 21.80 | 23.73 / 23.73 | 67.06 / 67.06 |
| `squelch` CQ15 | 28.03 / 28.03 | 28.03 / 28.03 | 28.03 / 28.03 | 28.03 | 28.53 / 28.53 | 21.02 / 21.02 | 71.05 / 71.05 |
| `mag` C32 | 2.57 / 3.52 | 1.82 / 2.77 | 1.04 / 1.64 | 1.06 | 0.66 / 1.27 | 1.45 / 2.02 | 6.86 / 8.53 |
| `mag` CQ15 | 5.09 / 21.27 | 5.09 / 11.02 | 2.25 / 2.64 | 2.26 | 1.22 / 2.64 | 3.26 / 4.52 | 9.03 / 18.28 |
| `agc` C32 | 23.79 / 24.03 | 23.04 / 24.03 | 17.39 / 24.03 | 17.41 | 16.12 / 24.03 | 17.61 / 17.03 | 64.61 / 64.05 |
| `agc` CQ15 | 72.86 / 72.86 | 72.86 / 72.86 | 69.40 / 69.40 | 72.86 | 70.31 / 70.31 | 54.77 / 54.77 | 141.14 / 141.14 |
| `dc_block_real` f32 | 11.51 / 11.51 | 11.51 / 11.51 | 8.76 / 8.76 | 11.51 | 8.76 / 8.76 | 16.01 / 16.01 | 28.53 / 28.53 |
| `dc_block_real` Q15 | 16.01 / 16.01 | 16.01 / 16.01 | 14.01 / 14.01 | 16.01 | 13.01 / 13.01 | 12.01 / 12.01 | 49.03 / 49.03 |
| `biquad` C32, 2 sections (vs KFR; CMSIS-DSP at `simd128`) | 17.43 / 81.59 | 17.37 / 76.59 | 13.17 / 53.61 | 13.18 | 13.23 / 72.00 | 12.61 / 46.63 | 43.04 / 118.83 |
| `biquad` C32: CMSIS-DSP stereo (information) | 57.54 | 57.54 | 45.04 |  | 45.04 | 41.53 |  |
| `biquad` CQ15, 2 sections (vs fallback) | 51.81 / 110.05 | 47.05 / 110.05 | 20.24 / 130.05 | 20.24 | 20.34 / 101.05 | 34.11 / 66.04 | 70.53 / 199.09 |
| `biquad` CQ15: CMSIS-DSP `df1_q15` on re and im (information) | 152.09 | 152.09 | 144.09 |  | 140.09 | 100.06 | 272.13 |
| `biquad_real` f32, 2 sections (vs KFR; CMSIS-DSP at `simd128`) | 14.69 / 40.80 | 14.26 / 38.30 | 13.24 / 26.81 | 13.25 | 11.87 / 36.00 | 12.75 / 23.32 | 40.09 / 57.94 |
| `biquad_real` f32: CMSIS-DSP `df2T_f32` (information) | 29.04 | 29.04 | 22.54 |  | 22.54 | 22.53 |  |
| `biquad_real` Q15, 2 sections (vs fallback) | 27.81 / 57.04 | 23.88 / 57.04 | 18.74 / 48.54 | 18.75 | 18.07 / 48.04 | 17.76 / 34.03 | 41.21 / 110.07 |
| `biquad_real` Q15: CMSIS-DSP `df1_q15` (information) | 76.04 | 76.04 | 72.04 |  | 70.04 | 50.03 | 136.07 |
| `biquad_fallback` C32, our fallback / CMSIS-DSP | 31.97 / 57.54 | 32.10 / 57.54 | 23.91 / 45.04 |  | 23.78 / 45.04 | 21.71 / 41.53 | 68.47 / 118.83 |
| `biquad_cq15_fallback`, our fallback / CMSIS-DSP | 110.05 / 152.09 | 110.05 / 152.09 | 130.05 / 144.09 |  | 101.05 / 140.09 | 66.04 / 100.06 | 199.09 / 272.13 |
| `biquad_real_fallback` f32, our fallback / CMSIS-DSP | 28.78 / 29.04 | 28.33 / 29.04 | 21.38 / 22.54 |  | 19.95 / 22.54 | 21.76 / 22.53 | 50.79 / 57.94 |
| `biquad_real_q15_fallback`, our fallback / CMSIS-DSP | 57.04 / 76.04 | 57.04 / 76.04 | 48.54 / 72.04 |  | 48.04 / 70.04 | 34.03 / 50.03 | 110.07 / 136.07 |
| `fft` C32, N = 64 | 15.06 / 18.88 | 15.06 / 25.30 | 5.84 / 5.88 | 6.28 | 5.28 / 5.53 | 10.52 / 18.28 | 35.50 / 63.23 |
| `fft` C32, N = 1024 | 25.16 / 25.56 | 25.16 / 33.81 | 10.33 / 11.33 | 10.37 | 5.57 / 11.07 | 15.14 / 25.85 | 58.57 / 91.17 |
| `fft` C32, N = 4096 | 28.93 / 31.17 | 28.93 / 39.44 | 11.45 / 13.27 | 11.49 | 5.92 / 11.81 | 17.37 / 30.85 | 67.64 / 105.58 |
| `fft` CQ15, N = 1024, block float | 28.51 / 233.83 | 22.26 / 233.83 | 10.19 / 233.83 | 10.24 | 5.80 / 233.83 | 15.79 / 179.33 | 48.92 / 460.87 |
| `fft` CQ15, N = 1024, fixed scaling | 25.27 / 233.83 | 19.87 / 233.83 | 9.11 / 233.83 | 9.18 | 5.29 / 233.83 | 12.79 / 179.33 | 43.57 / 460.87 |
| `fft` CQ15, N = 1024, fixed scaling, full scale | 27.07 / 233.83 | 20.81 / 233.83 | 9.61 / 233.83 | 9.65 | 5.44 / 233.83 | 13.83 / 179.33 | 46.16 / 460.87 |
| `rfft` f32, N = 1024 | 16.72 / 15.32 | 16.72 / 15.32 | 8.24 / 13.54 | 8.30 | 5.51 / 12.82 | 10.43 / 18.75 | 41.91 / 59.40 |

Accuracy (SNR in dB against the f64 reference), ours / comparison:

| Kernel | sse2 | sse4.2 | avx2 | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | exact | exact | exact | exact | exact | exact |
| `convert` CS16 to C32 | exact | exact | exact | exact | exact | exact |
| `convert` CU8 to C32 | exact | exact | exact | exact | exact | exact |
| `convert_out` C32 to CS16 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 |
| `convert_out` C32 to CS8 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 |
| `convert_out` C32 to CU8 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 | 47.3 / 47.3 |
| `rotate` C32 | 142.8 / 107.8 | 142.8 / 107.8 | 142.7 / 115.0 | 142.7 / 114.5 | 143.9 / 115.2 | 142.8 / 107.8 |
| `rotate` CQ15 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 | 91.2 / 91.2 |
| `fir` C32, 64 taps | 141.7 / 141.7 | 141.7 / 141.7 | 142.1 / 142.3 | 142.1 / 141.9 | 142.1 / 141.0 | 141.3 / 136.4 |
| `fir_decimate` C32, 127 taps, M = 10 | 141.0 / 139.0 | 141.0 / 139.0 | 142.0 / 141.2 | 141.9 / 141.1 | 141.8 / 138.5 | 141.0 / 134.3 |
| `fir_interpolate` C32, L = 2, 128 taps | 141.9 / 141.5 | 141.9 / 141.5 | 141.8 / 142.4 | 141.8 / 141.9 | 141.8 / 141.2 | 141.2 / 136.5 |
| `resample` C32, 3 / 2, 192 taps | 141.6 / 141.4 | 141.6 / 141.4 | 142.2 / 142.2 | 142.3 / 141.9 | 142.2 / 140.8 | 141.6 / 136.3 |
| `mag_sq` C32 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 |
| `mag_sq` CQ15 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 | 92.8 / 92.8 |
| `quad_demod` C32 | 143.5 / 144.2 | 143.5 / 144.2 | 143.5 / 126.5 | 143.5 / 126.5 | 143.5 / 126.5 | 143.5 / 143.6 |
| `quad_demod` CQ15 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 | 81.2 / 81.2 |
| `fm_mod` C32 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 |
| `squelch` C32 | exact | exact | exact | exact | exact | exact |
| `squelch` CQ15 | exact | exact | exact | exact | exact | exact |
| `mag` C32 | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.5 | 150.5 / 150.9 | 150.5 / 150.9 | 150.5 / 150.5 |
| `mag` CQ15 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 | 96.2 / 96.2 |
| `agc` C32 | 119.9 / 118.9 | 119.9 / 118.9 | 119.9 / 118.9 | 119.9 / 118.9 | 119.9 / 118.9 | 119.9 / 118.9 |
| `agc` CQ15 | exact | exact | exact | exact | exact | exact |
| `dc_block_real` f32 | 124.8 / 124.8 | 124.8 / 124.8 | 124.8 / 124.8 | 124.8 / 124.8 | 124.8 / 124.8 | 124.8 / 124.8 |
| `dc_block_real` Q15 | exact | exact | exact | exact | exact | exact |
| `biquad` C32, 2 sections (f64 model; KFR, CMSIS-DSP at `simd128`) | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 136.1 | 136.1 / 134.9 |
| `biquad` CQ15, 2 sections (f64 model; fallback) | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 | 71.6 / 71.6 |
| `biquad_real` f32, 2 sections (f64 model; KFR, CMSIS-DSP at `simd128`) | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 | 136.7 / 136.7 |
| `biquad_real` Q15, 2 sections (f64 model; fallback) | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 | 71.5 / 71.5 |
| biquad fallback entries (f64 model; CMSIS-DSP): C32, CQ15, f32, Q15 | 136.1 / 134.9, 71.6 / 56.7, 136.7 / 136.7, 71.5 / 56.6 | the same | the same | the same | the same | the same |
| `fft` C32, N = 64 | 142.1 / 140.2 | 142.1 / 142.1 | 140.7 / 141.4 | 140.7 / 141.4 | 141.7 / 142.3 | 142.5 / 141.6 |
| `fft` C32, N = 1024 | 138.7 / 137.9 | 138.7 / 138.7 | 139.1 / 139.1 | 139.1 / 139.1 | 138.8 / 139.1 | 138.2 / 138.9 |
| `fft` C32, N = 4096 | 137.8 / 137.2 | 137.8 / 137.8 | 138.2 / 137.9 | 138.2 / 137.9 | 138.0 / 138.3 | 137.4 / 137.7 |
| `fft` CQ15, N = 1024, block float | 65.3 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 |
| `fft` CQ15, N = 1024, fixed scaling | 47.9 / 43.5 | 46.7 / 43.5 | 46.7 / 43.5 | 46.7 / 43.5 | 48.0 / 43.5 | 46.7 / 43.5 |
| `fft` CQ15, N = 1024, fixed scaling, full scale | 60.0 / 55.1 | 58.9 / 55.1 | 58.9 / 55.1 | 58.9 / 55.1 | 59.9 / 55.1 | 58.9 / 55.1 |
| `rfft` f32, N = 1024 | 138.6 / 138.5 | 138.6 / 138.5 | 138.7 / 138.1 | 138.3 / 138.1 | 138.5 / 138.4 | 137.8 / 138.5 |

Notes.

- **Comparisons per level.** The `sse2` FFT is judged against FFTW out of place (D3). In place, FFTW needs 31.5 / 35.1 / 40.7 instructions per sample at N = 64 / 1024 / 4096 (ours 0.48 / 0.72 / 0.71 of that), and rustfft, which has no planner below SSE4.1, is scalar there (56.8 / 93.1 / 112.2). On `neon` FFTW (`--enable-neon`, out of place) is information: 15.9 / 26.8 / 39.7. The `CQ15` FFT is judged against KissFFT, whose fixed-point code is scalar on every platform; Ne10 int16 on `neon` needs 19.8 instructions per sample at 44.6 dB (ours 0.80 with block float, 0.65 with fixed scaling). The cheapest VOLK implementation on `neon` is the default dispatch for every kernel but `convert_cs16` (`neon`, 2.52), `fir`, `fir_decimate`, `fir_interpolate` and `resample` (`neon_unroll`, 150.0, 26.7, 169.5 and 171.4), `mag_sq` (`neon`, 1.52), `quad_demod` (19.3) and `convert_out` to `CU8` (`neon`, 4.02); ours is below all of them (`convert_out` to `CU8` 0.99 of the cheapest).
- **Run-time detection** (`avx2-runtime`) costs at most 0.02 times the instructions of the fixed `avx2` build for every accelerated kernel but `fft` C32 at N = 64 (1.07, 6.28 against 5.84 instructions per sample: about 28 instructions more per call, 18 of them in `level::dispatch_fn!`); that entry is `dev` (open item M13-9). Against rustfft it is 1.07 as well (5.88). The kernels that stay with the fallback run its baseline code in the run-time build and the AVX2 build of the same code at `avx2` (`info`: `squelch` C32 1.25, `agc` CQ15 1.05, `dc_block_real` f32 1.31 and Q15 1.14).
- **`convert_out`.** Rounds to nearest even like VOLK (decision D7) and keeps NaN as zero; that costs one compare and one AND per vector at SSE/AVX2 (VOLK does not look at NaN), and nothing on NEON and SIMD128. The `U8` encoding reproduces the f64 rounding exactly, which costs one byte bound per value more than VOLK at `sse2`, `sse4.2`, `avx2` and `neon` (a pack and two minimums on x86, a compare and a narrowing on NEON) and a floor and a compare at `avx512`. At `sse2` that is 1.35 times VOLK (open item M13-8); at `sse4.2` and `avx2` 1.05 and 1.06, within the release limit.
- **`rotate`.** The block oscillator needs 2.37 instructions per sample at `avx2` against 5.28 for VOLK 3.3 (and 3.07 for the VOLK 3.1 of Ubuntu 24.04) and is more accurate (142.7 dB against 115.0 dB). Its per-call cost (about 450 instructions at `avx2`) shows in short calls: 9.1 instructions per sample at n = 64 and 32.6 at n = 16.
- **`fir_interpolate`, `resample`.** Every branch of the polyphase filter runs as a contiguous filter in the forms of `fir` (`M = 1`) and `fir_decimate`, with its outputs written at the stride `L`; the comparison makes one VOLK dot product per output with the taps of its branch, as GNU Radio does. The counts are per output sample.
- **`rfft`.** The inner complex FFT of `N / 2` points is the backend's. The post-processing forms 4 bins and their mirrors per vector at `sse2`, `sse4.2`, `neon` and `simd128`; at `avx2` and `avx512` the fallback's loop, compiled at the level, is cheaper. At `sse2` and `sse4.2` the comparison is FFTW's real-input transform (15.32 instructions per sample at both levels): ours is 1.09 times that, within the release limit (the fallback's loop was 1.22 times). rustfft's real planner needs 50.96 at `sse2` (scalar) and 24.54 at `sse4.2`.
- **`fir`, `fir_decimate`.** The comparison is GNU Radio's `fir_filter_ccf` (one aligned VOLK dot product per output, D6). Ours is at most 0.2 dB below the comparison (`fir` at `avx2`, 142.1 against 142.3 dB) and above it elsewhere, within the 1 dB rule.
- **`quad_demod`.** At `sse2` and `sse4.2` VOLK has no SSE `atan2`; the comparison is the correctly rounded `generic` code (144.2 dB), 0.7 dB above ours, which is within the rule (decision D9).
- **FFT.** `C32` reads its twiddles from static per-span tables on x86-64 and AArch64 (feature `fft-tables`, [backends.md](backends.md)); wasm32 is measured without them (with them: 31.0 / 53.1 / 61.8 fuel per sample). The accuracy of the single-vector transform at N = 64 scatters by about 1.9 dB between levels (140.7 to 142.5 dB), with the comparisons between 140.2 and 142.3 dB. `CQ15`: the radix-4 transform is not bit-identical with the fallback (decision D10); on the bench input it is far more accurate than KissFFT (which rounds every stage of its fixed-point code).
- **`fm_mod`.** On wasm32 it is measured without the NCO tables (feature `nco-tables`, off there by default): 34.78 fuel per sample, 27.77 with the tables. At `avx512` it uses gathers (see the caveat at the top).
- **`squelch`.** Not accelerated (M19; [backends.md](backends.md)): the average power is a first-order recursion and the gate decision depends on it. Ours is the fallback, so every level is `info` with a ratio of 1.00 against the fallback, and the output is bit-identical (accuracy `exact`). The counts differ between levels only by code generation of the fallback (for example 21.84 at `avx2` against 27.35 at `sse2` and in the run-time-detection build, which runs the fallback compiled for the baseline target; that is why `avx2-runtime` is `info` for `squelch` too). Input of the measurement: noise with a burst in the middle third, so the gate opens and closes once per call; 4096 samples at 48 kHz, 2 ms time constant.
- **`mag`.** `C32` is the loop of `mag_sq` with a vector `sqrt` on the sums and needs 1.04 instructions per sample at `avx2` against 1.64 for `volk_32fc_magnitude_32f`; the generic code is the path on every level, and it is bit-exact (IEEE `sqrt` is correctly rounded). `CQ15` computes the rounded integer root from a single-precision estimate and one correction step in each direction, with the products from `pmaddwd` (`dot_i16x8` on SIMD128, `smlal` on NEON), and needs 5.09 / 5.09 / 2.25 / 1.22 / 3.26 / 9.03 instructions per sample at `sse2` / `sse4.2` / `avx2` / `avx512` / `neon` / `simd128`; `volk_16ic_magnitude_16i` needs 21.27 / 11.02 / 2.64 / 2.64 / 4.52 / 18.28 (on `neon` the cheapest combination of VOLK, the auto-vectorised `generic` code, needs 3.64, which is information: 0.90). The NEON path takes one correction step instead of two (a biased single-precision estimate that is never above the root, decisions log of M20; 4.39 before). VOLK scales to float and back and rounds differently from the integer rule: both are 96.2 dB against the unrounded magnitude. The entries are `release` at all six levels (every ratio is below 1.00; open item M20-3 closed). The result is bit-exact with the fallback on every level (the sweep over all 2^32 raw pairs, [testing.md](testing.md)).
- **`agc`, `dc_block_real`.** The gain of `agc` follows the output of the previous sample, and `dc_block_real` is a first-order recursion on the previous output (M20; [backends.md](backends.md)). `dc_block_real` and the `CQ15` `agc` are not accelerated and are `info` against the fallback at every level. The `C32` `agc` keeps the recursion scalar and computes the rest in vectors (next item). The input of `agc` is noise whose amplitude steps from 0.01 to 0.3 to 0.003 in thirds (the gain moves in both directions within a call; about 22 % of the samples are attacks, 37 % releases at or above `reference / 4` and 41 % releases below it), that of `dc_block_real` noise of 0.2 with an offset of 0.3.
- **`biquad`, `biquad_real` against KFR and CMSIS-DSP (D15 to D20).** The measured filter is the fourth-order Butterworth low-pass at 0.1 of the sample rate (two sections) on noise of the amplitude 0.3; the counts are per sample for both sections. On the maintainer's direction the SIMD entries are judged against KFR 7.2.0 (f32, C32; CMSIS-DSP at `simd128`, where KFR is not built) or our fallback (Q15, CQ15, which no library has in SIMD on our levels), and four fallback entries judge our fallback against CMSIS-DSP 1.18.0 generic C compiled for the same level. The six entries against a reference library are `release` and pass at every level; the two Q15 SIMD entries are `info`, because our own fallback is no fair gate for a SIMD branch (D21; the fallbacks themselves are gated against CMSIS-DSP). All of them are at or below the goal at every level (`biquad_real_fallback`, open item M22-9 until 2026-10-28, is now 0.88 to 0.99 of CMSIS-DSP's `df2T_f32`). The accuracy of every biquad entry is the SNR against the f64 model of the cascade (direct form I in `f64`, the coefficients as the kernels see them, no rounding between the sections), the same yardstick for both sides.
  - **Vector paths (D18).** All four types run a wavefront over the sections: two consecutive sections in the lanes of one vector, the second one sample behind the first, with the operations of the fallback per lane, so the output is the fallback's bit for bit; the `Q15` types compute exactly in `f64` lanes. The same generic `fearless_simd` code on every level, AVX-512 included (the AVX-512 special case of M22 and its 25 % rule are gone). Against KFR: `biquad_real` 0.33 (`avx512`) to 0.55 (`neon`), `biquad` `C32` 0.18 to 0.27; against CMSIS-DSP at `simd128` 0.69 and 0.36. Against our fallback: `Q15` 0.37 to 0.52, `CQ15` 0.16 to 0.52. Run-time detection costs at most 0.01 instructions per sample (1.00 at `avx2-runtime` for all four); the entries go through one function per level (`dispatch_fn!`), and the hot loops call methods instead of closures, because with run-time detection a closure is not compiled with the target features of the level (measured before the change: `biquad_real` `Q15` 26.2 against 19.9 at `avx2`).
  - **Fallback (D16, D17).** The `f32` states are flushed on a fixed schedule (once per 16 samples, D17) instead of after every sample, and `biquad_real` `f32` runs two sections per pass (each sample through both); with that the `f32` fallback is within 0.88 to 0.99 of CMSIS-DSP's `df2T_f32` (before: 1.59 to 2.04). In the blocks of sixteen samples of a pair, the flush of the second section (after its sample 14) moves to between the two steps of sample 15, beside the flush of the first section, and in the other fifteen samples the second section stores its output before its state update; both keep every operation and its order (M22-9: at `sse2`, `sse4.2` and `avx2` 30.26, 31.79 and 22.87 before, 28.78, 28.33 and 21.38 after against 29.04, 29.04 and 22.54; `avx512` 21.57 to 19.95; `neon` and `simd128` unchanged). The `C32` fallback is 0.52 to 0.58 of CMSIS-DSP's stereo cascade, the `Q15` fallback 0.67 to 0.81 and the `CQ15` fallback 0.66 to 0.90 of `df1_q15` (the rounding and saturation of the `Q15` fallback is now a clamp of the shifted sum, which compiles to conditional moves instead of branches).
  - **Accuracy (D20).** The `f32` association is now the one of KFR and CMSIS-DSP's `df2T_f32`, `s1 = (b1 x + s2) - a1 y`; ours and KFR then give the same SNR (`biquad_real` 136.7 dB, `biquad` 136.1 dB; with the earlier association ours was 1.1 dB lower and failed the 1 dB rule). CMSIS-DSP's stereo cascade keeps the other association (134.9 dB). CMSIS-DSP's Q15 cascade truncates and is 14.9 dB less accurate than ours (56.7 against 71.6 dB).
  - **Notes.** KFR's lane-shifting cascade is slower than CMSIS-DSP's scalar code for two sections at every level where it is built, `avx512` included (`biquad_real`: 40.80 against 29.04 instructions per sample at `sse2`, 26.81 against 22.54 at `avx2`, 36.00 against 22.54 at `avx512`, 23.32 against 22.53 at `neon`; KFR's `avx512` build counts more than its `avx2` build). KFR is the gate on the maintainer's direction (D15: the established SIMD library of the same algorithm class), but CMSIS-DSP is the stronger reference for these entries; every `f32` and `C32` SIMD entry also passes against it, which the results record as information next to KFR (`biquad_real` 0.49 to 0.59, the highest 0.59 at `avx2` and 0.57 at `neon`; `biquad` `C32` 0.29 to 0.30). KFR 7.2.0 loses its state on a call with fewer samples than sections (see the tests of `benches/kernels`); the measured calls have 64 samples and more. The counts were measured with Intel SDE 9.58 for `sse2`, `sse4.2` and `avx512`, callgrind for `avx2`, QEMU 10.2.1 with the `insn` plugin for `neon` and wasmtime fuel for `simd128`, first merged into `benches/results/` with `cargo xtask count --merge`; full runs of `cargo xtask count` on all four platforms (2026-10-27) confirmed every record unchanged and wrote KFR, CMSIS-DSP and `kfr_cxx` into the `references` section of the results files (CMSIS-DSP only on `wasm32`).
- **`agc` `C32` against GNU Radio.** The reference is the `agc_cc` loop of GNU Radio (member for member in `benches/refs/gr-agc.c`: the four members live in one object that the output stores may alias, so the gain is loaded and stored in every iteration, `sqrtf` keeps its `errno` rule and `_max_gain > 0.0` is a comparison in double precision, as in `agc.h`). It is compiled once per platform with the flags of GNU Radio's own build, because the loop is header-only code that VOLK's dispatch never touches: 24.03 instructions per sample on x86-64 at every level, 17.03 on aarch64 (GCC fuses the products and sums into `fmadd`, which our kernels do not do, and omits the `errno` branch), 64.05 fuel on wasm32 (clang). Ours: 23.79 (0.99) at `sse2`, 23.04 (0.96) at `sse4.2`, 17.39 (0.72) at `avx2`, 16.12 (0.67) at `avx512`, 17.61 (1.03) at `neon` and 64.61 (1.01) on `simd128`, all `release` within the limit of 1.10x (open item M20-6 closed; before: 23.84 / 23.84 / 20.84 / 20.84 / 21.84 / 62.21, `neon` `dev` at 1.28). The clamp of the gain to both limits in the attack and in the first release form (a gain at a limit could leave it by one rounding with a clamp on one side) costs 0.48 to 0.64 instructions per sample on x86-64, 0.58 at `neon` and 3.53 fuel on `simd128` (a compare and a select per limit there): 17.03 (1.00) at `neon` and 61.08 (0.95) on `simd128` with the one-sided clamp. How: the output level is taken as `g |x|` instead of `|g x|` (the same value, other roundings), so the magnitudes `|x|` and the products `g x` do not depend on the recursion; the SIMD backend computes them in vectors per block of 256 samples (2.25 and 2.0 instructions per sample at `neon`) and the scalar recursion in between costs a load, a multiplication, the update and an eighth of a store pair (12.5 at `neon` on this input). The release from `reference / 4` up is evaluated as `g + (r_r ref / |x| - r_r g)` (no division by the output level), the release below it as `g + g ((r_r / lo) (ref - m))` (no division), the attack as specified ([kernels.md](kernels.md)); the backend tells per block whether a magnitude is zero or NaN and leaves out the test of the hold otherwise (0.47 instructions per sample less at `neon` on this input, the compare and the `and` per vector included). On NEON the limits are `fminnm`/`fmaxnm` written as `asm!`: with `f32::min`/`max` the compiler adds an `fminnm` of every operand that comes from another basic block, and with the intrinsics `vminnm_f32` it evaluates every branch and selects (31.12 instructions per sample); the recursion runs eight samples per iteration there (four on x86-64, whose 16 registers do not hold eight), written out because the compiler keeps a loop of eight steps (22.01 with that loop). No multiply-add (decision D5). The fallback alone (the information line) is one plain scalar loop for every target since open item M20-7 (no target-specific code in the loop, no `black_box`; the square root `num::sqrt_f32` reaches the hardware instruction through lane 0 of `f32x4_sqrt` on wasm32 with `simd128`, as before the review; four samples per iteration, the limits a compare and a select, the two-sided clamps taking the minimum with `g_max` first so that the three branches do not end in the same operation, which the compiler would otherwise move behind them and turn into a branch of its own on x86-64): 28.24 at `sse2` and `sse4.2`, 23.36 at `avx2` and `avx512`, 22.42 at `neon`, 64.11 fuel on `simd128` and 76.16 on Cortex-M4 (before the review 23.84 / 20.84 / 21.84 / 62.21 / 80.50; the review's version with `black_box` 27.87 / 22.73 / 22.92 / 66.89 / 75.52). The level `g |x|` costs the multiplication and, because `x` is now used twice (for `g x` and for its squares), two register copies per sample at `sse2` and two loads at `avx2` that the loads folded into the products avoided before; in the old form the output was squared in place. The loop of GNU Radio does less than ours: no lower gain limit, no NaN or silence rule, no bounded update, and an additive rule that is level dependent (22 dB low after 5 time constants for a step of 30 dB). The accuracy is 119.9 dB against the f64 model of the specification at every level (123.1 dB before; a constant `1 - r_r` rounded to f32 in the release, which would save one instruction, gave 103 dB and was not taken). On wasm32 `libm` computes the square root in software (the instruction of `core` is unstable there), 189.51 fuel per sample before `sdrk_core::num::sqrt_f32` took the `simd128` instruction. The `CQ15` fallback does not divide in integers: each quotient of the update is an f64 estimate followed by one exact correction in 64-bit integers (the remainder `n - k d` as the wrapping difference of the low 64 bits, which bounds on the estimate keep below `2^63`; the argument is in the comments of `fallback/agc.rs`), and the rounded square root is one f64 square root (exact up to `2^32`). That gives 72.86 instructions per sample at `sse2` and `sse4.2`, 69.40 at `avx2`, 70.31 at `avx512`, 54.77 at `neon`, 141.14 fuel on wasm32 and 1333.88 on Cortex-M4 (counted since M20-7), against 124.61 / 124.61 / 125.08 / 110.27 / 111.35 / 620.79 with the 64-bit divisors and the `u128` division of the attack (software `__udivti3`; before them, with `i128` divisions, 190.78 at `neon` and 968.79 fuel). On x86-64 the estimate is turned into an integer by adding `2^52` (the saturating `as` costs about five instructions more there), on AArch64, wasm32 and Cortex-M4 by `as` (one instruction or operator on the first two); the `avx2-runtime` build of this fallback is the baseline code and counts 72.86 against 69.40 of `avx2` (`info`). The values are those of `Reference` bit for bit (unit tests against plain `u128` division at the edges of the domain, at 2 million random points and next to multiples of the divisor, and a kernel test over the parameter extremes). The update divides by the output level and costs more instructions than the former logistic update: the choice is correctness (a step of 30 dB settles in 5 time constants, the loop cannot collapse, silence holds the gain). At `sse2` and on wasm32 the limits are a compare and a select (`f32::max`/`min` cost 16 instructions per sample more at `sse2` in an isolated copy of the loop, measured with callgrind, and 27.8 more fuel on wasm32).
- **AVX-512** counts come from SDE between the SSC markers and are exact; an earlier version of the method without markers varied by up to 0.07 instructions per sample.
- **Variants.** The misaligned and small-block lines of the results files are information; they show the per-call cost of the conversions (for example `convert` CS16 at `avx512`, n = 64: 2.63 against 2.08 for VOLK).

## Pipeline benchmark

`cargo xtask count --pipeline` runs `examples/flows/fm_receiver.json` (a `cu8` input at 2.4 MHz, the receiver of the examples with its spectrum output) in blocks of 4096 input samples under callgrind (x86-64 with AVX2, differential method) and writes `benches/results/pipeline.json`:

| Measure | Value |
| --- | --- |
| instructions per input sample | 461.2 (545 before the SIMD optimization) |
| latency of `audio` (`Pipeline::latency`) | 30.14 samples of 48 kHz, 627.9 µs (group delay 27.14 samples, up to 3 samples buffered) |

At 2.4 Msps this is about 1.1 G instructions per second of signal.

The same command also counts the SSB chain of M21 (`ddc` at +300 kHz with decimation 50, 4 kHz cutoff and 2 kHz transition, then `agc` with its defaults, then `ssb_demod` USB with its defaults: the receiver of `examples/flows/ssb_receiver.json` without the squelch, at the rates of the end-to-end SSB test), in the same blocks of 4096 input samples; the line is in `chains.ssb` of `benches/results/pipeline.json` (information, no limit):

| Chain | Instructions per input sample | Latency of `audio` |
| --- | --- | --- |
| `ddc` + `agc` + `ssb_demod` | 631.4 | 481.52 samples of 48 kHz, 10031.7 µs (group delay 479.52 samples: 436 from the 873-tap audio filter, the rest from the channel filter; up to 2 samples buffered) |

`ssb_demod` adds no kernel. It calls `rotate`, `fir_decimate` and a second `rotate` once per piece of 256 samples, never per sample, so every call goes through the facade to the vector paths of the backend (the check is `the_block_equals_the_kernel_chain` in `crates/blocks/sdrk-blocks/tests/ssb_demod.rs`, which compares the block with the three kernel calls bit for bit; the differential count above includes the block, and the drift of the FM line from 461.2 to 461.0 is the toolchain of the day). The channel filter of the chain is the same `ddc` as in the FM line, so the difference of the two lines (170 instructions per input sample) is the cost of `agc` and `ssb_demod` at 48 kHz amortised over the 50 input samples per audio sample.

## Running the measurements

```sh
cargo xtask count --platform x86_64          # needs valgrind and Intel SDE (SDRK_SDE=<path to sde64>)
cargo xtask count --platform x86_64-avx512   # needs Intel SDE
cargo xtask count --platform aarch64         # needs a QEMU user-mode build with plugins and the aarch64 cross compilers
cargo xtask count --platform wasm32          # needs wasmtime and clang with the wasi-libc headers
cargo xtask count --pipeline                 # needs valgrind
```

Options: `--level L` and `--kernel NAME` (repeatable) restrict the run (the results file is written only for a complete run), `--no-write` keeps `benches/results` unchanged, `--no-variants` skips the informational misaligned and small-block runs, and `--strict` fails for every entry that is not a clean `release` entry (the release check of open item M13-4).

Tools.

- **The C references** need `curl`, `sha256sum`, `tar`, `make`, `cmake`, `ninja` and `python3` with `mako` (VOLK), and for `aarch64` and `wasm32` the cross compilers (`gcc-aarch64-linux-gnu`, `g++-aarch64-linux-gnu`; `clang` and `llvm-ar` with the wasi-libc headers). `benches/refs/build.sh` returns at once when the prefix already holds a build of the same script with the same compilers. Environment: `SDRK_REFS_DOWNLOADS` (download cache), `SDRK_REFS_JOBS` (default 2).
- **Intel SDE** is pinned in CI to version 9.58.0 (`https://downloadmirror.intel.com/859732/sde-external-9.58.0-2025-06-16-lin.tar.xz`, sha256 `f849acecad4c9b108259c643b2688fd65c35723cd23368abe5dd64b917cc18c0`) and cached; it is proprietary freeware of Intel and is not part of the repository.
- **QEMU** must be configured with `--enable-plugins`; the packaged `qemu-aarch64` of the distributions is not. CI builds QEMU 10.2.1 (`configure --target-list=aarch64-linux-user --enable-plugins`, then `ninja qemu-aarch64`) in the nightly job. Point `SDRK_QEMU_AARCH64` at the binary. xtask builds the `insn` plugin into `target/tools/libinsn.so` if it is missing, from the QEMU tree named by `SDRK_QEMU_SRC` or from the two files `tests/tcg/plugins/insn.c` and `include/qemu/qemu-plugin.h` that it downloads for the tag `SDRK_QEMU_TAG` (default `v10.2.1`; the location of the plugin source has moved between QEMU releases). Needs `gcc`, `pkg-config` and `libglib2.0-dev`. `SDRK_AARCH64_SYSROOT` (default `/usr/aarch64-linux-gnu`, from `libc6-dev-arm64-cross`) is the sysroot of the target libc.
- **wasmtime** with fuel (`-W fuel=N`).
- For `aarch64-unknown-linux-gnu` cross builds xtask sets `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc`.

## CI

The count jobs run in a pinned container image (`ubuntu:26.04` by digest) with the packages of a pinned apt snapshot, so that the compilers of the C references do not change under the counts; the references are cached by the build script, the image and the snapshot.

- `count` counts `x86_64` on every pull request that touches `crates/core`, `crates/backends`, `crates/facade`, `benches` or `xtask`, with `--no-variants --no-write`: the misaligned and small-block variants are information and are not judged.
- On a pull request that touches kernels, backends, `benches`, `xtask` or the toolchain, `count_x64` measures `x86_64` without the variants. In the nightly run `count_x64` measures `x86_64` with the variants and the pipeline benchmark, and `count_x64_linux_sde_avx512`, `count_arm64_linux` and `count_wasm32` the platforms `x86_64-avx512`, `aarch64` and `wasm32` (`tree.yml`, [testing.md](testing.md)). They fail only when `cargo xtask count` fails (a limit, an accuracy check or a measurement error); the difference of the fresh results against the committed `benches/results` is shown for information (`git diff`) and the fresh files are kept as an artifact (`count-results-<job>`). The counts drift with the runner (the CPU name in `host_cpu`, and for `x86_64` the counts themselves, because the pool mixes EPYC 7763, 9V45 and 9V74) and with the `stable` toolchain of the day (for example 0.01 instructions per sample in wasm32), so a byte-for-byte comparison fails without a change of the code. Instruction counts are deterministic for one machine and one toolchain. `xtask` removes every `QEMU_*` variable from the environment of `qemu-aarch64`, because qemu-user reads them as options (`QEMU_VERSION` is `-version`).
- wasmtime is installed from its GitHub release tarball, pinned by version and SHA-256 (`WASMTIME_SHA256`).

## Cortex-M4 (fallback, report only)

`cargo xtask count --platform cortex-m4` counts fifteen kernels of the portable fallback on a Cortex-M4F in `qemu-system-arm` (the `insn` plugin) and writes `benches/results/cortex-m4.json`; the nightly job `count_m4_qemu` runs it. The method is the differential one above, with the workload binary `count` of `targets/qemu-cortex-m4`; there are no limits and no comparison library yet (CMSIS-DSP comes with the `arm-dsp` backend). Details in [embedded.md](embedded.md#instruction-counts-cortex-m4). Note that the M4F has a single-precision FPU only: `convert_out_cs16` rounds in `f64` (R7) and so counts about a thousand instructions per sample in software floating point.
