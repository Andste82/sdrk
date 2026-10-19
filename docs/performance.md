# Performance

Performance is measured in executed instructions per sample in an emulator and compared with an established library (VOLK, rustfft, CMSIS-DSP, ESP-DSP) at equal accuracy against the f64 reference.

| Instructions per sample vs. reference library | Meaning |
| --- | --- |
| ≤ 1.00× | goal: equal or better |
| ≤ 1.10× | accepted |
| ≤ 1.15× | hard limit; CI fails above it |
| ≤ 1.50× | allowed only while a kernel is under development, never in a release |

Instruction counts are a strong proxy on in-order cores (Cortex-M, ESP32) and a weaker one on out-of-order CPUs and for memory-bound kernels. They say nothing about cache behaviour, the cost of one instruction (a gather is one instruction and many cycles) or the clock; they are the portable, reproducible measure of how much work a kernel does.

## Method

`cargo xtask count --platform <platform>` builds the measurement binary of `benches/kernels` for the platform and runs it under an instruction counter.

- **Differential count.** `count run <kernel> <impl> <k>` does one warm-up call and then `k` calls on the same buffers. It is started twice, with `k = 0` and with `k = K`, and the instructions per sample are `(count(K) - count(0)) / (K * samples)`. Process start, allocation, set-up (the plan of rustfft, the VOLK dispatcher) and the warm-up call cancel. `K = 16` calls of 4096 samples; a kernel with fewer samples per call (the FFT sizes 64 and 1024) gets proportionally more calls, so every measurement covers 65 536 samples and the difference stays far above the run-to-run noise of the process.
- **Inputs.** PCG32 noise with the seed `1000 + index of the kernel`, 4096 samples (`N`), buffers aligned to 64 bytes for both sides. The FFT is measured in place on the same buffer; its values grow with every call, which does not change the number of instructions.
- **Counters.**

| Platform | Counter | Levels |
| --- | --- | --- |
| `x86_64` | Valgrind `callgrind` (Linux; not available on Windows, so the count job runs on Linux only) | `sse2` (default build), `sse4.2` (`-C target-cpu=x86-64-v2`), `avx2` (`-C target-cpu=x86-64-v3`), and `avx2-runtime` (default build with run-time detection, informational) |
| `x86_64-avx512` | Intel SDE, `sde64 -spr -mix` (the `*total` line of the mix report) | `avx512` (`-C target-cpu=sapphirerapids`) |
| `aarch64` | `qemu-aarch64` in user mode with the `insn` plugin (`-plugin libinsn.so -d plugin`) | `neon` |
| `wasm32` | wasmtime with fuel: the smallest fuel with which the run completes, found by bisection | `simd128` |

  `iai-callgrind` is not used: it needs a runner binary installed in the version of the library and adds nothing to the raw `callgrind` total that xtask reads.
- **Comparisons.** On x86-64 the pairs of the table below (VOLK 3.3.0, which the count jobs build from the release tarball with a pinned checksum so that the comparison does not depend on the distribution's version; locally the distribution's 3.3.0 gives the same numbers; rustfft with `FftPlanner`, forward, in place). `fm_mod` and the `CQ15` FFT have no library; they are compared with our own fallback. On AArch64 and wasm32 there is no VOLK, so every kernel but the `C32` FFT (rustfft) is compared with our own fallback in the same binary. For the level `sse2` the VOLK dispatcher is restricted by a `volk_config` that xtask writes (`benches/target/volk-sse/volk/volk_config`) to the implementations that need no more than SSE3 (the generic C code where VOLK has none); for `avx2` and `avx2-runtime` a `volk_config` (`benches/target/volk-avx2/volk/volk_config`) restricts it to AVX2/FMA, because on a runner with AVX-512 VOLK would otherwise dispatch `rotator2`, `atan2` and `dot_prod` to their AVX-512 implementations and the comparison would depend on the CPU; at `avx512` it picks its best implementation itself, as our kernels use AVX-512 there too.
- **Accuracy.** `count accuracy <kernel> <impl>` prints the SNR of one run against the f64 reference. A kernel fails if its SNR is more than 1 dB below the SNR of its comparison.
- **VOLK is LGPL-3.0.** It is linked dynamically from the system by `benches/volk-sys` (hand-written `extern "C"` declarations; VOLK declares its kernels as function-pointer variables that its dispatcher fills in) and is never a dependency of a library crate. `benches/` is a Cargo workspace of its own that is not a member of the repository workspace (`exclude = ["benches"]`), so `cargo deny` does not see it.

### Where VOLK has no implementation for a level

VOLK has no SSE implementation of `rotator2` and `atan2`, and none below SSE4.1 of `volk_8i_s32f_convert_32f`. The only candidate is then the portable `generic` code, and that is not SSE code: VOLK compiles it once per instruction set and uses the copy of the best machine, which GCC has auto-vectorised with AVX2 (the library contains twelve copies of `volk_8i_s32f_convert_32f_generic`, four of them with `ymm` registers). Comparing our SSE2 build with it is not a comparison at the same instruction set.

- The level `sse4.2` compares at SSE4.2 where VOLK has an implementation for it: `convert` CS8 and CS16 against the SSE4.1 versions, the rest as at `sse2`.
- The level `sse2` is still measured, but the entries whose reference is `generic` (`convert_cs8`, `rotate`, `quad_demod`) are `info` in `limits.json`: reported with their numbers, never judged. `rotate` and `quad_demod` are `info` at `sse4.2` too, because VOLK has no SSE version of them.

## Limits

`benches/limits.json` names, per kernel and level, `{ "status": "release" | "dev" }`. `cargo xtask count` writes `benches/results/<platform>.json` (counts, ratio, SNR of ours and of the comparison) and fails if a `release` entry exceeds **1.15×** or a `dev` entry exceeds **1.50×**; between 1.10× and 1.15× it prints a warning. A kernel that is not listed counts as `dev`. A `release` entry is set only where the measured ratio is at most 1.10×, so that a different version of VOLK on the runner does not fail the job at once.

An entry may carry a `"waiver"` text that names the open item in `docs/open-items.md` which tracks a limit that is not met yet. A waived exceedance is printed (`waived`) and recorded in the results file, but does not fail the command; the numbers are never hidden. The unit tests of the limit logic (`xtask/src/count.rs`) include a deliberately slow dummy kernel with status `release`, which makes the command fail.

## Results

Instructions per sample of our kernel divided by those of the comparison (lower is better); `dev` marks an entry whose status is `dev`, `waived` one with a waiver. All other entries are `release`. Measured with VOLK 3.3 and rustfft 6.4 on the development machine; the CI runner may differ in the version of VOLK.

| Kernel | Compared with (x86-64) | sse2 | avx2 | avx512 | neon | simd128 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | `volk_8i_s32f_convert_32f` on 2N values | 0.30 | 0.90 | 0.96 | 0.03 | 0.04 |
| `convert` CS16 to C32 | `volk_16i_s32f_convert_32f` on 2N values | 1.06 | 0.93 | 0.97 | 0.03 | 0.04 |
| `convert_out` C32 to CS16 | `volk_32f_s32f_convert_16i` on 2N values | 2.31 waived | 1.99 waived | 2.25 waived | 0.06 | 0.08 |
| `rotate` C32 | `volk_32fc_s32fc_x2_rotator2_32fc` | 0.53 | 1.23 dev | 0.52 | 0.46 | 0.94 |
| `fir` C32, 64 taps | `volk_32fc_32f_dot_prod_32fc` per output | 1.11 dev | 0.71 | 0.52 | 0.51 | 0.24 |
| `fir_decimate` C32, 127 taps, M = 10 | `volk_32fc_32f_dot_prod_32fc` per output | 1.33 dev | 0.94 | 0.94 | 0.50 | 0.26 |
| `mag_sq` C32 | `volk_32fc_magnitude_squared_32f` | 1.39 dev | 1.00 | 0.45 | 1.29 dev | 1.37 dev |
| `quad_demod` C32 | `volk_32fc_x2_multiply_conjugate_32fc` + `volk_32fc_s32f_atan2_32f` | 0.36 | 0.86 | 0.52 | 0.27 | 0.40 |
| `fm_mod` C32 | our fallback | 0.78 | 0.22 | 0.07 | 0.45 | 0.40 |
| `fft` C32, N = 64 | `rustfft` | 10.36 waived | 5.22 waived | 4.92 waived | 1.78 waived | 0.91 waived |
| `fft` C32, N = 1024 | `rustfft` | 6.15 waived | 2.54 waived | 1.59 waived | 1.46 dev | 0.59 |
| `fft` C32, N = 4096 | `rustfft` | 5.65 waived | 2.27 waived | 1.50 dev | 1.30 dev | 0.53 |
| `fft` CQ15, N = 1024 | our fallback | 0.55 | 0.37 | 0.40 | 0.42 | 0.41 |

Instructions per sample, ours / comparison:

| Kernel | sse2 | avx2 | avx2, run-time detection | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | 4.15 / 14.02 | 1.59 / 1.77 | 1.60 / 1.77 | 0.92 / 0.96 | 2.26 / 77.03 | 8.04 / 184.06 |
| `convert` CS16 to C32 | 4.28 / 4.02 | 1.65 / 1.77 | 1.66 / 1.77 | 0.87 / 0.90 | 2.50 / 99.03 | 10.54 / 242.06 |
| `convert_out` C32 to CS16 | 9.28 / 4.02 | 4.28 / 2.14 | 4.28 / 2.14 | 2.09 / 0.93 | 8.02 / 124.01 | 23.29 / 308.03 |
| `rotate` C32 | 17.02 / 32.11 | 6.52 / 5.31 | 6.66 / 5.31 | 2.65 / 5.09 | 12.28 / 26.51 | 33.29 / 35.29 |
| `fir` C32, 64 taps | 250.00 / 225.00 | 144.43 / 203.00 | 144.44 / 203.00 | 113.62 / 217.01 | 212.54 / 413.42 | 623.71 / 2646.13 |
| `fir_decimate` C32, 127 taps, M = 10 | 54.63 / 41.15 | 37.55 / 39.85 | 37.67 / 39.84 | 41.99 / 44.80 | 46.19 / 91.54 | 143.28 / 544.12 |
| `mag_sq` C32 | 3.51 / 2.52 | 1.51 / 1.52 | 1.52 / 1.52 | 0.72 / 1.62 | 2.26 / 1.76 | 11.03 / 8.03 |
| `quad_demod` C32 | 55.52 / 154.04 | 9.40 / 10.92 | 9.41 / 10.92 | 3.49 / 6.71 | 29.53 / 110.51 | 76.29 / 190.90 |
| `fm_mod` C32 | 48.42 / 61.91 | 13.65 / 60.91 | 13.66 / 61.91 | 3.87 / 56.89 | 19.02 / 41.94 | 43.53 / 109.89 |
| `fft` C32, N = 64 | 62.64 / 6.05 | 31.06 / 5.95 | 32.42 / 5.99 | 27.45 / 5.57 | 32.34 / 18.17 | 118.47 / 130.11 |
| `fft` C32, N = 1024 | 69.39 / 11.28 | 28.83 / 11.33 | 30.00 / 11.27 | 17.66 / 11.08 | 37.67 / 25.84 | 134.18 / 226.47 |
| `fft` C32, N = 4096 | 73.97 / 13.10 | 30.13 / 13.28 | 31.46 / 13.10 | 17.89 / 11.92 | 40.12 / 30.86 | 144.36 / 274.79 |
| `fft` CQ15, N = 1024 | 327.73 / 594.80 | 204.81 / 549.51 | 206.62 / 594.80 | 211.12 / 529.15 | 173.59 / 415.51 | 485.17 / 1195.50 |

Accuracy (SNR in dB against the f64 reference), ours / comparison, `exact` where both are bit-exact with the reference:

| Kernel | sse2 | avx2 | avx512 | neon | simd128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `convert` CS8 to C32 | exact | exact | exact | exact | exact |
| `convert` CS16 to C32 | exact | exact | exact | exact | exact |
| `convert_out` C32 to CS16 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 | 95.5 / 95.5 |
| `rotate` C32 | 140.8 / 108.8 | 141.1 / 115.0 | 141.1 / 114.5 | 141.1 / 133.0 | 140.8 / 133.0 |
| `fir` C32, 64 taps | 141.8 / 142.3 | 142.3 / 142.6 | 142.3 / 142.3 | 141.8 / 136.3 | 141.8 / 136.3 |
| `fir_decimate` C32, 127 taps, M = 10 | 140.5 / 139.5 | 141.5 / 139.0 | 141.8 / 141.2 | 140.5 / 133.3 | 140.5 / 133.3 |
| `mag_sq` C32 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 | 149.0 / 149.0 |
| `quad_demod` C32 | 144.2 / 144.2 | 141.8 / 126.5 | 141.8 / 126.5 | 144.2 / 143.6 | 144.2 / 143.6 |
| `fm_mod` C32 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 | 133.2 / 133.2 |
| `fft` C32, N = 64 | 141.0 / 141.4 | 140.9 / 141.4 | 141.7 / 141.4 | 142.2 / 142.3 | 141.0 / 142.1 |
| `fft` C32, N = 1024 | 138.0 / 139.1 | 138.4 / 139.1 | 138.7 / 139.1 | 138.4 / 139.1 | 138.0 / 138.8 |
| `fft` C32, N = 4096 | 137.4 / 137.9 | 137.7 / 137.9 | 137.7 / 137.9 | 137.7 / 138.3 | 137.4 / 137.7 |
| `fft` CQ15, N = 1024 | 53.9 / 53.9 | 53.9 / 53.9 | 53.9 / 53.9 | 53.9 / 53.9 | 53.9 / 53.9 |

Notes.

- **Levels.** `neon` and `simd128` are compared with our own fallback (except the FFT), which the compiler vectorises where the loop allows (the squares of `mag_sq` on NEON). Where the comparison is VOLK at `avx2`, VOLK's own kernels are hand-written intrinsics.
- **Run-time detection** costs nothing measurable per call: the `avx2-runtime` column is within 0.15 instruction per sample of `avx2` for every kernel but the two FFTs (1.2 to 1.8 more per sample, another inlining of the same code).
- **`convert_out`.** The specified rounding is `floor(v + 0.5)` with NaN as zero; VOLK rounds to nearest even and does not look at NaN. Making the ties come out right costs about as many instructions again as the conversion itself (waiver, open item M13-1).
- **`rotate` at `avx2`.** The polynomial oscillator needs 6.5 instructions per sample against 5.3 for the recurrence of VOLK. The recurrence is not a function of the phase alone, so its result depends on the block size, and it is less accurate (115 dB against our 141 dB).
- **`fir`/`fir_decimate` at `sse2`.** The vectors have two complex samples; the VOLK variant of the same level reduces with SSE3 shuffles.
- **FFT.** `C32`: the in-place transform without per-stage twiddle tables and without scratch needs two to five times the instructions of rustfft on x86-64, 1.3 to 1.8 times on NEON and fewer on wasm32 (where rustfft is scalar). Waiver, open item M13-2. At N = 64 on wasm32 the accuracy is 1.05 dB below rustfft (limit 1 dB; waiver, open item M13-3).

## Pipeline benchmark

`cargo xtask count --pipeline` runs `examples/flows/fm_receiver.json` (a `cu8` input at 2.4 MHz, the receiver of the examples with its spectrum output) in blocks of 4096 input samples under callgrind (x86-64 with AVX2, differential method) and writes `benches/results/pipeline.json`:

| Measure | Value |
| --- | --- |
| instructions per input sample | 545 |
| latency of `audio` (`Pipeline::latency`) | 30.14 samples of 48 kHz, 627.9 µs (group delay 27.14 samples, up to 3 samples buffered) |

At 2.4 Msps this is about 1.3 G instructions per second of signal.

## Running the measurements

```sh
cargo xtask count --platform x86_64          # needs valgrind, libvolk-dev, pkg-config
cargo xtask count --platform x86_64-avx512   # needs Intel SDE; SDRK_SDE=<path to sde64>
cargo xtask count --platform aarch64         # needs a QEMU user-mode build with plugins
cargo xtask count --platform wasm32          # needs wasmtime
cargo xtask count --pipeline                 # needs valgrind
```

Tools.

- **Intel SDE** is pinned in CI to version 9.58.0 (`https://downloadmirror.intel.com/859732/sde-external-9.58.0-2025-06-16-lin.tar.xz`, sha256 `f849acecad4c9b108259c643b2688fd65c35723cd23368abe5dd64b917cc18c0`) and cached; it is proprietary freeware of Intel and is not part of the repository.
- **QEMU** must be configured with `--enable-plugins`; the packaged `qemu-aarch64` of the distributions is not. CI builds QEMU 10.2.1 (`configure --target-list=aarch64-linux-user --enable-plugins`, then `ninja qemu-aarch64`) in the nightly job. Point `SDRK_QEMU_AARCH64` at the binary. xtask builds the `insn` plugin into `target/tools/libinsn.so` if it is missing, from the QEMU tree named by `SDRK_QEMU_SRC` or from the two files `tests/tcg/plugins/insn.c` and `include/qemu/qemu-plugin.h` that it downloads for the tag `SDRK_QEMU_TAG` (default `v10.2.1`; the location of the plugin source has moved between QEMU releases). Needs `gcc`, `pkg-config` and `libglib2.0-dev`. `SDRK_AARCH64_SYSROOT` (default `/usr/aarch64-linux-gnu`, from `libc6-dev-arm64-cross`) is the sysroot of the target libc.
- **wasmtime** with fuel (`-W fuel=N`).
- For `aarch64-unknown-linux-gnu` cross builds xtask sets `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc` (package `gcc-aarch64-linux-gnu`).

## CI

`count` runs `cargo xtask count --platform x86_64` on every pull request that touches `crates/core`, `crates/backends`, `crates/facade`, `benches` or `xtask`; the nightly run adds the platforms `x86_64-avx512`, `aarch64` and `wasm32`, the pipeline benchmark, and checks that the committed results are up to date. Instruction counts are deterministic, so the committed `benches/results/*.json` change only when the code, the toolchain, VOLK or rustfft change.

## Cortex-M4 (fallback, report only)

`cargo xtask count --platform cortex-m4` counts eleven kernels of the portable fallback on a Cortex-M4F in `qemu-system-arm` (the `insn` plugin) and writes `benches/results/cortex-m4.json`; the nightly job `count-embedded` runs it. The method is the differential one above, with the workload binary `count` of `targets/qemu-cortex-m4`; there are no limits and no comparison library yet (CMSIS-DSP comes with the `arm-dsp` backend). Details in [embedded.md](embedded.md#instruction-counts-cortex-m4). Note that the M4F has a single-precision FPU only: `convert_out_cs16` rounds in `f64` (R7) and so counts about a thousand instructions per sample in software floating point.
