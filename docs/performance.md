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
| rustfft | 6.4 (Cargo dependency of `benches/kernels`) | MIT OR Apache-2.0 | all; `wasm_simd` on wasm32 (target-specific) | the `C32` FFT reference except at `sse2` |

The libraries are linked only into the measurement binary of `benches/kernels`, which is built on the measuring machine and never distributed; none of them is a dependency of a library crate of SDRk. `benches/` is a Cargo workspace of its own that is not a member of the repository workspace (`exclude = ["benches"]`), so `cargo deny` does not see it. VOLK is linked dynamically (`benches/volk-sys`, hand-written `extern "C"` declarations; VOLK declares its kernels as function-pointer variables that its dispatcher fills in), FFTW, KissFFT and Ne10 statically through `benches/cref-sys`.

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
| `quad_demod` C32 | `volk_32fc_x2_multiply_conjugate_32fc` + `volk_32fc_s32f_atan2_32f` |
| `rotate` CQ15, `mag_sq` CQ15, `quad_demod` CQ15, `fm_mod` C32, `squelch` C32 and CQ15 | our fallback (no established library) |
| `fft` C32, N = 64, 1024, 4096 | FFTW at `sse2`, rustfft (`FftPlanner`, forward, in place) elsewhere |
| `fft` CQ15, N = 1024, block float and fixed scaling (noise of 8000 LSB, and of full scale for fixed scaling) | KissFFT `FIXED_POINT=16`; Ne10 int16 as information on `neon` |
| `rfft` f32, N = 1024 | FFTW's real-input transform (`fftwf_plan_dft_r2c_1d`, out of place) at `sse2` and `sse4.2` (FFTW has no codelets above SSE2, so it is the stronger reference there too), rustfft's real-input planner (`realfft` 3.5, forward, out of place) elsewhere; the other of the two and our fallback as information (FFTW also on `neon`) |

VOLK's dispatcher is pinned per level by a `volk_config` that xtask writes (`benches/target/<level>/volk-pinned/volk/volk_config`), and the measured processes run without `VOLK_GENERIC` and without the variables that override single kernels:

- `sse2`, `sse4.2`: the defaults of VOLK's `sse2_64_mmx` and `sse4_2_64_mmx` machines, which the emulated CPUs select.
- `avx2`, `avx2-runtime`: the implementations that need no more than AVX2 and FMA, so that a runner with AVX-512 does not dispatch `rotator2`, `atan2` and `dot_prod` to AVX-512 code.
- `avx512`: VOLK's own choice, as our kernels use AVX-512 there too.
- `neon`: the default dispatch of VOLK's `neonv8` machine (D11), pinned so that a change of VOLK's ranking shows up as a missing implementation instead of a silently different reference. Every VOLK kernel also gets the cheapest combination of VOLK implementations of the machine as information (`goal: cheapest VOLK implementation`).
- `simd128`: VOLK has no wasm32 machine; its `generic` protokernels compiled for wasm32 with `simd128` are what its dispatcher would take. They gate `mag_sq` and the conversions only (D12); for `rotate`, the FIR family and `quad_demod` they are scalar code and the entries are `info`.

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
| `fft` C32, N = 64 | 142.1 / 140.2 | 142.1 / 142.1 | 140.7 / 141.4 | 140.7 / 141.4 | 141.7 / 142.3 | 142.5 / 141.6 |
| `fft` C32, N = 1024 | 138.7 / 137.9 | 138.7 / 138.7 | 139.1 / 139.1 | 139.1 / 139.1 | 138.8 / 139.1 | 138.2 / 138.9 |
| `fft` C32, N = 4096 | 137.8 / 137.2 | 137.8 / 137.8 | 138.2 / 137.9 | 138.2 / 137.9 | 138.0 / 138.3 | 137.4 / 137.7 |
| `fft` CQ15, N = 1024, block float | 65.3 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 | 63.5 / 43.7 |
| `fft` CQ15, N = 1024, fixed scaling | 47.9 / 43.5 | 46.7 / 43.5 | 46.7 / 43.5 | 46.7 / 43.5 | 48.0 / 43.5 | 46.7 / 43.5 |
| `fft` CQ15, N = 1024, fixed scaling, full scale | 60.0 / 55.1 | 58.9 / 55.1 | 58.9 / 55.1 | 58.9 / 55.1 | 59.9 / 55.1 | 58.9 / 55.1 |
| `rfft` f32, N = 1024 | 138.6 / 138.5 | 138.6 / 138.5 | 138.7 / 138.1 | 138.3 / 138.1 | 138.5 / 138.4 | 137.8 / 138.5 |

Notes.

- **Comparisons per level.** The `sse2` FFT is judged against FFTW out of place (D3). In place, FFTW needs 31.5 / 35.1 / 40.7 instructions per sample at N = 64 / 1024 / 4096 (ours 0.48 / 0.72 / 0.71 of that), and rustfft, which has no planner below SSE4.1, is scalar there (56.8 / 93.1 / 112.2). On `neon` FFTW (`--enable-neon`, out of place) is information: 15.9 / 26.8 / 39.7. The `CQ15` FFT is judged against KissFFT, whose fixed-point code is scalar on every platform; Ne10 int16 on `neon` needs 19.8 instructions per sample at 44.6 dB (ours 0.80 with block float, 0.65 with fixed scaling). The cheapest VOLK implementation on `neon` is the default dispatch for every kernel but `convert_cs16` (`neon`, 2.52), `fir`, `fir_decimate`, `fir_interpolate` and `resample` (`neon_unroll`, 150.0, 26.7, 169.5 and 171.4), `mag_sq` (`neon`, 1.52), `quad_demod` (19.3) and `convert_out` to `CU8` (`neon`, 4.02); ours is below all of them (`convert_out` to `CU8` 0.99 of the cheapest).
- **Run-time detection** (`avx2-runtime`) costs at most 0.02 times the instructions of the fixed `avx2` build for every kernel but `fft` C32 at N = 64 (1.07, 6.28 against 5.84 instructions per sample: about 28 instructions more per call, 18 of them in `level::dispatch_fn!`); that entry is `dev` (open item M13-9). Against rustfft it is 1.07 as well (5.88).
- **`convert_out`.** Rounds to nearest even like VOLK (decision D7) and keeps NaN as zero; that costs one compare and one AND per vector at SSE/AVX2 (VOLK does not look at NaN), and nothing on NEON and SIMD128. The `U8` encoding reproduces the f64 rounding exactly, which costs one byte bound per value more than VOLK at `sse2`, `sse4.2`, `avx2` and `neon` (a pack and two minimums on x86, a compare and a narrowing on NEON) and a floor and a compare at `avx512`. At `sse2` that is 1.35 times VOLK (open item M13-8); at `sse4.2` and `avx2` 1.05 and 1.06, within the release limit.
- **`rotate`.** The block oscillator needs 2.37 instructions per sample at `avx2` against 5.28 for VOLK 3.3 (and 3.07 for the VOLK 3.1 of Ubuntu 24.04) and is more accurate (142.7 dB against 115.0 dB). Its per-call cost (about 450 instructions at `avx2`) shows in short calls: 9.1 instructions per sample at n = 64 and 32.6 at n = 16.
- **`fir_interpolate`, `resample`.** Every branch of the polyphase filter runs as a contiguous filter in the forms of `fir` (`M = 1`) and `fir_decimate`, with its outputs written at the stride `L`; the comparison makes one VOLK dot product per output with the taps of its branch, as GNU Radio does. The counts are per output sample.
- **`rfft`.** The inner complex FFT of `N / 2` points is the backend's. The post-processing forms 4 bins and their mirrors per vector at `sse2`, `sse4.2`, `neon` and `simd128`; at `avx2` and `avx512` the fallback's loop, compiled at the level, is cheaper. At `sse2` and `sse4.2` the comparison is FFTW's real-input transform (15.32 instructions per sample at both levels): ours is 1.09 times that, within the release limit (the fallback's loop was 1.22 times). rustfft's real planner needs 50.96 at `sse2` (scalar) and 24.54 at `sse4.2`.
- **`fir`, `fir_decimate`.** The comparison is GNU Radio's `fir_filter_ccf` (one aligned VOLK dot product per output, D6). Ours is at most 0.2 dB below the comparison (`fir` at `avx2`, 142.1 against 142.3 dB) and above it elsewhere, within the 1 dB rule.
- **`quad_demod`.** At `sse2` and `sse4.2` VOLK has no SSE `atan2`; the comparison is the correctly rounded `generic` code (144.2 dB), 0.7 dB above ours, which is within the rule (decision D9).
- **FFT.** `C32` reads its twiddles from static per-span tables on x86-64 and AArch64 (feature `fft-tables`, [backends.md](backends.md)); wasm32 is measured without them (with them: 31.0 / 53.1 / 61.8 fuel per sample). The accuracy of the single-vector transform at N = 64 scatters by about 1.9 dB between levels (140.7 to 142.5 dB), with the comparisons between 140.2 and 142.3 dB. `CQ15`: the radix-4 transform is not bit-identical with the fallback (decision D10); on the bench input it is far more accurate than KissFFT (which rounds every stage of its fixed-point code).
- **`fm_mod`.** On wasm32 it is measured without the NCO tables (feature `nco-tables`, off there by default): 34.78 fuel per sample, 27.77 with the tables. At `avx512` it uses gathers (see the caveat at the top).
- **`squelch`.** Not accelerated (M19; [backends.md](backends.md)): the average power is a first-order recursion and the gate decision depends on it. Ours is the fallback, so every level is `info` with a ratio of 1.00 against the fallback, and the output is bit-identical (accuracy `exact`). The counts differ between levels only by code generation of the fallback (for example 21.84 at `avx2` against 27.35 at `sse2` and in the run-time-detection build, which runs the fallback compiled for the baseline target; that is why `avx2-runtime` is `info` for `squelch` too). Input of the measurement: noise with a burst in the middle third, so the gate opens and closes once per call; 4096 samples at 48 kHz, 2 ms time constant.
- **AVX-512** counts come from SDE between the SSC markers and are exact; an earlier version of the method without markers varied by up to 0.07 instructions per sample.
- **Variants.** The misaligned and small-block lines of the results files are information; they show the per-call cost of the conversions (for example `convert` CS16 at `avx512`, n = 64: 2.63 against 2.08 for VOLK).

## Pipeline benchmark

`cargo xtask count --pipeline` runs `examples/flows/fm_receiver.json` (a `cu8` input at 2.4 MHz, the receiver of the examples with its spectrum output) in blocks of 4096 input samples under callgrind (x86-64 with AVX2, differential method) and writes `benches/results/pipeline.json`:

| Measure | Value |
| --- | --- |
| instructions per input sample | 461.2 (545 before the SIMD optimization) |
| latency of `audio` (`Pipeline::latency`) | 30.14 samples of 48 kHz, 627.9 µs (group delay 27.14 samples, up to 3 samples buffered) |

At 2.4 Msps this is about 1.1 G instructions per second of signal.

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

`cargo xtask count --platform cortex-m4` counts eleven kernels of the portable fallback on a Cortex-M4F in `qemu-system-arm` (the `insn` plugin) and writes `benches/results/cortex-m4.json`; the nightly job `count_m4_qemu` runs it. The method is the differential one above, with the workload binary `count` of `targets/qemu-cortex-m4`; there are no limits and no comparison library yet (CMSIS-DSP comes with the `arm-dsp` backend). Details in [embedded.md](embedded.md#instruction-counts-cortex-m4). Note that the M4F has a single-precision FPU only: `convert_out_cs16` rounds in `f64` (R7) and so counts about a thousand instructions per sample in software floating point.
