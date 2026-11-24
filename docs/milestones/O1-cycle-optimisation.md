# O1 — Cycle optimisation round: the entries slower than their library

> Plan: [M13.1](M13.1-cycles-only.md) (cycles per sample as the speed measure, the like-with-like statuses of D2764 to D2779) · Open items it works on: I13.1-7, I13.1-8, I13.1-9, I13.1-10, I24-3 ([open-items.md](../open-items.md)) · Next: [BACKLOG](BACKLOG.md)

> **Status: in progress** (planned 2026-11-24 on the maintainer's order of the same day; branch `opt-round-cycles` from `main` 6274f880). Decision numbers D2800 to D2899, open items IO1-1 to IO1-n. No seeds.

## Goal

Make SDRk at least as fast as its foreign reference library, in simulated cycles per sample on both core models of a level, wherever it is slower today. The maintainer's order (2026-11-24), by the numbers of the review list of that day:

1. **Item 7:** the entries above `_limits.release` 1.10 of their library on at least one model: the fallbacks of I13.1-7, the SIMD entries of I13.1-8, `stereo_pll_fallback` (I13.1-10) and `stereo_matrix_fallback` (I24-3).
2. **Item 2, together with item 7:** the entries above 1.50 that waited for the maintainer's decision (D2779) are optimised as well (`am_mod_fallback` at `sse2`, `sse4.2` and `neon`; `fir_decimate_cq15_fallback` at `sse2`, `sse4.2` and `neon`; `stereo_matrix_fallback` at `avx2`). For `am_mod_fallback` it is checked first whether liquid-dsp does the same work; where it does less, that is reported with evidence instead of forcing the ratio.
3. **Item 8 (I13.1-9):** `fir_cq15`, `fir_interpolate_cq15` and `fir_interpolate_real_q15` in the run-time dispatch build (`avx2-runtime`) run the code of the fixed `avx2` build (as D2711 did for `scramble` and `mag_sq`), outputs bit-identical. Today `znver3` / `alderlake`: `fir_cq15` 84.10 / 98.96 against 40.62 / 61.15, `fir_interpolate_cq15` 97.70 / 110.63 against 52.45 / 71.49, `fir_interpolate_real_q15` 60.92 / 67.03 against 42.13 / 38.06 cycles per sample (the hotfix D65 chooses the grouped fallback engine per compiled level, which the run-time build does not take).
4. **Item 9:** real SIMD128 references at `simd128`: VOLK's SSE protokernels built for wasm32 through Emscripten's SSE headers (`system/include/compat/*intrin.h`, as for KFR) or SIMDe, for `convert_*`, `add*`, `mul*`, `mag*`, `interleave` / `deinterleave`, `stereo_matrix` and the like; those entries are judged against them (`release` / `dev` / `info` by the rules), and the reference build records its SIMD level.

Not in this round (a parallel branch, `refs-fair-measure`): FFTW with AVX, Ne10 at `neon`, the reference-level check, the full CI measurement, `docs/performance.md` and the CI cache.

## Rules

- **Speed** is cycles per sample (`llvm-mca` with the corrections of D942), on the reference model and the counter-model of the level: `sse2`, `sse4.2` `skylake` / `znver2`; `avx2`, `avx2-runtime` `znver3` / `alderlake`; `avx512` `znver4` / `sapphirerapids`; `neon` `cortex-a72` / `cortex-a76`. Instructions only at `simd128` and on Cortex-M4. A branch costs its cycles plus mispredictions times the penalty of the core (D1611, D1612; `skylake` 20, `znver2` 16, `cortex-a72` 20).
- **Judged against a foreign library**, like with like per level; our own code against our own code is information only (`cargo xtask check cycles`, D2767 to D2779). `release`: at most 1.10 of the library on both models; `dev`: at most 1.50, with an open item.
- **Accuracy first (D34):** the f64 golden is the optimum; ours at most 1 dB below the library; no accuracy for speed without the maintainer (an accuracy trade only at least 3 dB better, library below 100 dB, at most 1.20 on both models). Numeric contracts (bit-exact classes, chunk invariance, no FMA where the contract says so, the radix-2 order of the `C32` FFT) do not change without the maintainer: such a part stops with an open item.
- **Fallbacks prefer Arm:** a fallback change is judged also on the AArch64 scalar build (`neon`), and the Cortex-M4 instructions per sample (`cargo xtask count --platform cortex-m4`) must not rise. Auto-vectorisation gains are welcome, never a reason to reject.
- No test, limit or bound is loosened, skipped or commented out. The CI container is the reference environment of committed numbers (`nightly.yml`, `measure_only`); locally only short checks of a few entries.

## Targets

From `benches/results/cycles/*.json` and `benches/limits.json` at 6274f880 (the full CI measurement of D2745 and the re-runs of D2762): every `dev` entry above 1.10 of its library on a model. Cycles per sample; a ratio is ours divided by the library on the same model. SNR 999 is an exact result.

### A. FFT fallbacks (I13.1-7)

| Entry | Level | Ours (cycles per sample, ref. / counter model) | Library (cycles per sample, ref. / counter model) | Ratio (ref. / counter model) | Models (ref. / counter) | SNR in dB (ours / library) | Library | Above 1.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fft_1024_fallback` | `sse2` | 31.44 / 29.67 | 26.16 / 28.91 | 1.20 / 1.03 | skylake / znver2 | 137.96 / 138.73 | CMSIS-DSP `arm_cfft_f32` | |
| `fft_1024_fallback` | `sse4.2` | 31.38 / 29.98 | 25.79 / 28.60 | 1.22 / 1.05 | skylake / znver2 | 137.96 / 138.73 | CMSIS-DSP `arm_cfft_f32` | |
| `fft_4096_fallback` | `sse2` | 37.59 / 35.67 | 30.53 / 32.90 | 1.23 / 1.08 | skylake / znver2 | 137.36 / 137.71 | CMSIS-DSP `arm_cfft_f32` | |
| `fft_4096_fallback` | `sse4.2` | 37.53 / 35.98 | 30.53 / 32.90 | 1.23 / 1.09 | skylake / znver2 | 137.36 / 137.71 | CMSIS-DSP `arm_cfft_f32` | |
| `fft_64_fallback` | `sse2` | 20.78 / 19.54 | 15.15 / 17.76 | 1.37 / 1.10 | skylake / znver2 | 141.01 / 140.0 | CMSIS-DSP `arm_cfft_f32` | |
| `fft_64_fallback` | `sse4.2` | 20.71 / 19.86 | 15.17 / 17.78 | 1.37 / 1.12 | skylake / znver2 | 141.01 / 140.0 | CMSIS-DSP `arm_cfft_f32` | |
| `rfft_1024_fallback` | `sse2` | 16.75 / 16.47 | 14.25 / 15.04 | 1.18 / 1.09 | skylake / znver2 | 137.86 / 138.31 | CMSIS-DSP `arm_rfft_fast_f32` | |
| `rfft_1024_fallback` | `sse4.2` | 16.63 / 16.16 | 13.39 / 14.58 | 1.24 / 1.11 | skylake / znver2 | 137.86 / 138.31 | CMSIS-DSP `arm_rfft_fast_f32` | |

The branch costs of D1612 widen these gaps: ours mispredicts 0.28 (`fft_64`), 0.15 (`fft_1024`), 0.14 (`fft_4096`) and 0.07 (`rfft_1024`) times per sample, CMSIS-DSP 0.03 to 0.01; with the penalty `fft_64_fallback` at `sse2` costs 26.40 / 24.04 against 15.78 / 18.27 (1.67 / 1.32), and at `avx2` (`release` in cycles, 1.003 / 1.04) 16.00 / 18.33 against 12.92 / 13.10 (1.24 / 1.40).

### B. SIMD entries (I13.1-8)

| Entry | Level | Ours (cycles per sample, ref. / counter model) | Library (cycles per sample, ref. / counter model) | Ratio (ref. / counter model) | Models (ref. / counter) | SNR in dB (ours / library) | Library | Above 1.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `convert_out_cu8` | `sse2` | 1.59 / 1.55 | 1.27 / 1.27 | 1.25 / 1.23 | skylake / znver2 | 47.27 / 47.27 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_cu8` | `sse4.2` | 1.57 / 1.52 | 1.33 / 1.39 | 1.18 / 1.09 | skylake / znver2 | 47.27 / 47.27 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_cu8` | `avx2` | 0.53 / 0.67 | 0.54 / 0.60 | 0.99 / 1.11 | znver3 / alderlake | 47.27 / 47.27 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_cu8` | `avx2-runtime` | 0.53 / 0.66 | 0.54 / 0.60 | 0.98 / 1.10 | znver3 / alderlake | 47.27 / 47.27 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_cu8` | `avx512` | 0.62 / 0.52 | 0.51 / 0.59 | 1.20 / 0.88 | znver4 / sapphirerapids | 47.27 / 47.27 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_real_rs8` | `avx2-runtime` | 0.23 / 0.29 | 0.25 / 0.26 | 0.90 / 1.10 | znver3 / alderlake | 47.24 / 47.24 | VOLK `volk_32f_s32f_convert_8i` | |
| `convert_out_real_ru8` | `sse2` | 0.81 / 0.79 | 0.64 / 0.64 | 1.26 / 1.24 | skylake / znver2 | 47.25 / 47.25 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_real_ru8` | `sse4.2` | 0.80 / 0.78 | 0.67 / 0.70 | 1.19 / 1.11 | skylake / znver2 | 47.25 / 47.25 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_real_ru8` | `avx2` | 0.28 / 0.35 | 0.28 / 0.30 | 1.00 / 1.14 | znver3 / alderlake | 47.25 / 47.25 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_real_ru8` | `avx2-runtime` | 0.27 / 0.34 | 0.28 / 0.30 | 0.99 / 1.12 | znver3 / alderlake | 47.25 / 47.25 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_real_ru8` | `avx512` | 0.32 / 0.27 | 0.26 / 0.30 | 1.21 / 0.90 | znver4 / sapphirerapids | 47.25 / 47.25 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `convert_out_real_ru8` | `neon` | 1.47 / 1.06 | 1.34 / 1.08 | 1.10 / 0.98 | cortex-a72 / cortex-a76 | 47.25 / 47.25 | VOLK `volk_32f_s32f_x2_convert_8u` | |
| `mul` | `avx512` | 0.58 / 0.39 | 0.49 / 0.54 | 1.18 / 0.71 | znver4 / sapphirerapids | 150.12 / 150.12 | VOLK `volk_32fc_x2_multiply_32fc` | |
| `scramble_c32_b8` | `avx512` | 0.58 / 2.66 | 2.36 / 2.41 | 0.25 / 1.11 | znver4 / sapphirerapids | 147.72 / 147.72 | lookup + VOLK `volk_32fc_x2_multiply_32fc` | |

The `avx2-runtime` ratios on `alderlake` are 1.1005 (`convert_out_cu8`), 1.12 (`convert_out_real_ru8`) and 1.1007 (`convert_out_real_rs8`).

### C. Other fallbacks and the entries above 1.50 (I13.1-7, I13.1-10, I24-3; item 2)

| Entry | Level | Ours (cycles per sample, ref. / counter model) | Library (cycles per sample, ref. / counter model) | Ratio (ref. / counter model) | Models (ref. / counter) | SNR in dB (ours / library) | Library | Above 1.50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `am_mod_fallback` | `sse2` | 1.09 / 1.16 | 0.59 / 0.72 | 1.84 / 1.61 | skylake / znver2 | 150.8 / 150.8 | liquid-dsp `ampmodem_modulate_block` | yes |
| `am_mod_fallback` | `sse4.2` | 1.09 / 1.16 | 0.59 / 0.72 | 1.84 / 1.61 | skylake / znver2 | 150.8 / 150.8 | liquid-dsp `ampmodem_modulate_block` | yes |
| `am_mod_fallback` | `avx2` | 0.65 / 0.47 | 0.53 / 0.54 | 1.22 / 0.87 | znver3 / alderlake | 150.8 / 150.8 | liquid-dsp `ampmodem_modulate_block` | |
| `am_mod_fallback` | `neon` | 2.91 / 1.52 | 1.28 / 1.03 | 2.27 / 1.48 | cortex-a72 / cortex-a76 | 150.8 / 152.07 | liquid-dsp `ampmodem_modulate_block` | yes |
| `deemphasis_fallback` | `sse2` | 15.09 / 12.09 | 13.15 / 17.11 | 1.15 / 0.71 | skylake / znver2 | 145.36 / 151.98 | GNU Radio `fm_deemph` | |
| `deemphasis_fallback` | `sse4.2` | 15.09 / 11.09 | 13.15 / 17.11 | 1.15 / 0.65 | skylake / znver2 | 145.36 / 151.98 | GNU Radio `fm_deemph` | |
| `deinterleave_cq15_fallback` | `avx2` | 0.23 / 0.23 | 0.20 / 0.26 | 1.14 / 0.86 | znver3 / alderlake | 999.0 / 999.0 | VOLK `volk_16ic_deinterleave_16i_x2_generic` | |
| `dot_cq15_fallback` | `sse2` | 2.20 / 2.53 | 1.78 / 2.03 | 1.23 / 1.24 | skylake / znver2 | 82.2 / 82.2 | CMSIS-DSP `arm_dot_prod_q15` | |
| `dot_cq15_fallback` | `sse4.2` | 2.03 / 1.72 | 2.04 / 1.49 | 1.00 / 1.15 | skylake / znver2 | 82.2 / 82.2 | CMSIS-DSP `arm_dot_prod_q15` | |
| `fir_decimate_cq15_fallback` | `sse2` | 30.40 / 45.20 | 24.93 / 25.79 | 1.22 / 1.75 | skylake / znver2 | 999.0 / 82.19 | CMSIS-DSP `arm_fir_decimate_q15` | yes |
| `fir_decimate_cq15_fallback` | `sse4.2` | 30.40 / 45.20 | 27.95 / 22.79 | 1.09 / 1.98 | skylake / znver2 | 999.0 / 82.19 | CMSIS-DSP `arm_fir_decimate_q15` | yes |
| `fir_decimate_cq15_fallback` | `avx2` | 21.61 / 22.05 | 42.02 / 17.52 | 0.51 / 1.26 | znver3 / alderlake | 999.0 / 82.19 | CMSIS-DSP `arm_fir_decimate_q15` | |
| `fir_decimate_cq15_fallback` | `neon` | 29.57 / 27.63 | 16.40 / 16.90 | 1.80 / 1.63 | cortex-a72 / cortex-a76 | 999.0 / 82.19 | CMSIS-DSP `arm_fir_decimate_q15` | yes |
| `fir_interpolate_cq15_fallback` | `sse2` | 137.49 / 207.95 | 137.61 / 164.43 | 1.00 / 1.26 | skylake / znver2 | 999.0 / 80.09 | CMSIS-DSP `arm_fir_interpolate_q15` | |
| `fir_interpolate_cq15_fallback` | `sse4.2` | 137.49 / 207.95 | 137.61 / 164.43 | 1.00 / 1.26 | skylake / znver2 | 999.0 / 80.09 | CMSIS-DSP `arm_fir_interpolate_q15` | |
| `fir_interpolate_real_fallback` | `neon` | 82.83 / 62.42 | 67.65 / 76.22 | 1.22 / 0.82 | cortex-a72 / cortex-a76 | 136.52 / 136.42 | CMSIS-DSP `arm_fir_interpolate_f32` | |
| `fir_interpolate_real_q15_fallback` | `sse2` | 74.97 / 112.89 | 68.81 / 82.21 | 1.09 / 1.37 | skylake / znver2 | 999.0 / 80.35 | CMSIS-DSP `arm_fir_interpolate_q15` | |
| `fir_interpolate_real_q15_fallback` | `sse4.2` | 74.97 / 112.89 | 68.81 / 82.21 | 1.09 / 1.37 | skylake / znver2 | 999.0 / 80.35 | CMSIS-DSP `arm_fir_interpolate_q15` | |
| `fir_interpolate_short_fallback` | `neon` | 30.89 / 23.59 | 27.28 / 29.90 | 1.13 / 0.79 | cortex-a72 / cortex-a76 | 143.09 / 143.04 | CMSIS-DSP `arm_fir_interpolate_f32` | |
| `mag_sq_fallback` | `avx512` | 0.37 / 0.26 | 0.31 / 0.27 | 1.21 / 0.94 | znver4 / sapphirerapids | 149.02 / 149.02 | CMSIS-DSP `arm_cmplx_mag_squared_f32` | |
| `mul_fallback` | `avx512` | 0.77 / 0.78 | 0.53 / 0.79 | 1.47 / 0.98 | znver4 / sapphirerapids | 148.5 / 148.5 | CMSIS-DSP `arm_cmplx_mult_cmplx_f32` | |
| `mul_real_q15_fallback` | `avx2` | 0.28 / 0.39 | 0.30 / 0.31 | 0.92 / 1.25 | znver3 / alderlake | 91.62 / 85.59 | CMSIS-DSP `arm_mult_q15` | |
| `stereo_matrix_fallback` | `sse2` | 1.55 / 1.67 | 1.07 / 1.38 | 1.45 / 1.21 | skylake / znver2 | 151.28 / 151.28 | CMSIS-DSP `arm_add_f32` | |
| `stereo_matrix_fallback` | `sse4.2` | 1.52 / 1.46 | 1.07 / 1.38 | 1.42 / 1.06 | skylake / znver2 | 151.28 / 151.28 | CMSIS-DSP `arm_add_f32` | |
| `stereo_matrix_fallback` | `avx2` | 0.63 / 0.86 | 0.60 / 0.53 | 1.06 / 1.61 | znver3 / alderlake | 151.28 / 151.28 | CMSIS-DSP `arm_add_f32` | yes |
| `stereo_pll_fallback` | `sse2` | 88.81 / 77.78 | 71.63 / 63.55 | 1.24 / 1.22 | skylake / znver2 | 57.02 / 9.46 | liquid-dsp `nco_crcf` | |
| `stereo_pll_fallback` | `sse4.2` | 88.29 / 77.34 | 71.63 / 63.55 | 1.23 / 1.22 | skylake / znver2 | 57.02 / 9.46 | liquid-dsp `nco_crcf` | |

Not targets (above 1.10 on a model, but `release` with an accuracy trade of D2760 or `info`): `fft_cq15_1024_default_fallback` at `sse2` (1.12 / 0.97), `halfband_decimate_real_q15_fallback` at `neon` (1.16 / 1.05), `quad_demod_cq15_fallback` at `avx512` (0.94 / 1.18), `stereo_pll_fallback` at `avx2` (1.11 / 1.02), and the `info` rows (another work or no library of the level).

## Causes and plan

Found with short `llvm-mca` looks at the loops (the corrected tool, stand-alone copies of the loops compiled with the toolchain of the branch) and the branch counts of the committed records. Every change keeps the bits of its contract; each step is measured in CI on both models of every level it touches, the fallbacks also on `neon` and Cortex-M4.

### A. FFT fallbacks

- **Cause.** (1) The bit reversal of x86 and wasm32 (`bit_reverse` in `fallback/fft.rs`) swaps under the data-independent but irregular test `i < j`: about one misprediction every 4 samples at N = 64 (0.28 per sample, 5.6 cycles of cost on `skylake`), CMSIS-DSP permutes from a table without that branch. (2) The butterflies: CMSIS-DSP runs radix-8 / radix-4 butterflies (fewer loads, stores and twiddle reads per sample), ours runs radix-2 in pairs of stages on x86 (D170: triples spilled the 16 vector registers). The radix-2 order is the numeric contract of the `C32` FFT (docs/kernels.md: "radix-2 in f32 without FMA"), so the arithmetic stays as it is.
- **Plan.** A1: the bit reversal without the data-dependent branch: visit only the pairs `i < rev(i)` by construction (the even `i < N/2` scheme of D170 that runs off x86, with the carried reversed counter of x86), or swap with a select. Same permutation, same bits. A2: the stage loops of small N (N = 64: six stages, the per-stage set-up and the first single stage of odd `log2 N`) with fewer twiddle reads (the first two stages have the twiddles 1 and -j, which need no product but give the same bits: `x * 1` and the swap with a sign are exact). A3: `rfft_1024` gains from A1 and A2 through its complex half; its split step is checked separately. Target: 1.10 on both models at `sse2` and `sse4.2`; `neon` and `avx2` must not get slower. If A1 to A3 leave `fft_64` above 1.10, an open item names the remaining gap and the radix question for the maintainer.

### B. SIMD entries

- **`convert_out_cu8`, `convert_out_real_ru8` (all x86 levels, `neon`), `convert_out_real_rs8` (`avx2-runtime`).** VOLK's loop per 4 floats: `mulps`, `addps` (the bias 128), `minps`, `maxps`, `cvtps2dq` and the packs (`volk_32f_s32f_x2_convert_8u_u_sse2`). Ours adds the NaN rule (NaN as 0: a compare and an `and`; VOLK's `minps` turns NaN into 255), the exact rounding of the `f64` formula of the contract (`rint(x 128 + 127.5)` in `f64`: the shift by `C = 0.5 - 2^-25` and the bound byte of the tiny negative values, whose masks are packed beside the values) and a scalar head up to the 16-byte boundary. Same SNR (47.27 dB both), so the extra work buys the NaN rule and the bit-exact rounding. Plan: B1 shorten the bound path (take the bound from the sign of `w = y + 2^-47` with an integer shift and fold it into the value before the pack, instead of a second pack chain of masks); B2 at `avx512` try the 256-bit form of `avx2` under the AVX-512 token (`znver4` splits 512-bit vectors into two halves; `sapphirerapids` has room: 0.52 against 0.59); B3 at `avx2-runtime` the same code as the fixed `avx2` build (it is within 1 % today; check for the dispatch overhead per call). If the remaining gap is the extra work of the contract, the evidence (instructions and ports per vector of both loops) goes into an open item instead of a forced change.
- **`mul` at `avx512` (`znver4` 1.18).** The split form of `mul_split_blocks` on 512-bit vectors; VOLK's row at `avx512` shows the figures of its `avx2` row (0.49 / 0.54; which protokernel runs is the reference-level check of the parallel branch). Our `avx2` form simulates 0.45 / 0.51 on `znver3` / `alderlake`. Plan: B4 run the `avx2` form (`vfmaddsub`, 256 bits) or the 512-bit `vfmaddsub` form at `avx512` and keep the faster on both models (D326 measured 0.51 against 0.47 for the 512-bit forms on `znver4`; the `sapphirerapids` figure must stay at most 0.54 x 1.10).
- **`scramble_c32_b8` at `avx512` (`sapphirerapids` 1.11).** `bits = 8` has 256 table phasors, more than `AVX512_PERMUTE_MAX`, so `c32_avx512_gather` reads them with `vgatherdps`, which the `sapphirerapids` model makes slow (2.66 against 0.58 on `znver4`). The reference fills a buffer by scalar lookups and multiplies with VOLK. Plan: B5 replace the gather by 64-bit scalar loads of the phasor pairs (`movq` + inserts, or a stack buffer of 16 pairs) and keep the planar product; `znver4` has room (0.58 against 2.36).

### C. Other fallbacks and the entries above 1.50

- **`am_mod_fallback` (`sse2`, `sse4.2` 1.84 / 1.61; `neon` 2.27; `avx2` 1.22): liquid-dsp does less work.** Its loop is `y = x * mod_index + 1` (no clamp, no NaN rule, no carrier product; `benches/refs/liquid-ampmodem.c`): per 4 samples `mulps`, `addps` and the interleave with zero. Ours computes `carrier * (1 + depth * clamp(x, -1, 1))` with NaN as 0 (D586): `maxps`, `minps`, `cmpordps`, `andps`, `mulps`, `addps`, `mulps` and the same interleave, 7 vector operations on the floating-point ports against 2. `llvm-mca` of the two loops at `sse2`: ours 4.34 cycles per 4 samples on `skylake` (bound on ports 0, 1 and 5), liquid 2.35; `znver2` 4.61 against 2.84. On AArch64 the clamp compiles to `fcmgt` + `bsl` pairs and the loop is bound on the two floating-point pipes of `cortex-a72` (2.89 cycles per sample); a form with `fmaxnm` / `fminnm` simulated 2.79, no real gain. The carrier product cannot be folded into the depth (`carrier + (carrier depth) x` rounds differently, the order is the contract). Plan: C1 report this with the loop listings as an open item (the ratio follows from the extra work of the contract, accepted by D586), after one more try of a shorter select of the clamp and the NaN rule on each target; no forced change.
- **`fir_decimate_cq15_fallback` (`sse2`, `sse4.2` up to 1.98 on `znver2`; `neon` 1.80 / 1.63; `avx2` 1.26 on `alderlake`).** The shared engine computes `NARROW_GROUP` = 8 outputs per group with 16 `i64` accumulators and 8 window pointers: on x86-64 (16 integer registers) half the accumulators live on the stack, every tap does 8 read-modify-write additions in memory and reloads the window pointers. `llvm-mca` of the loop: 1.17 cycles per multiply-add on `skylake`, 1.44 on `znver2` (the measured 30.40 / 45.20 show the same lean to `znver2`). The `fir_interpolate_cq15` and `fir_interpolate_real_q15` fallbacks (`znver2` 1.26 and 1.37; `WIDE_GROUP` = 16 accumulators for the real type) share the cause. On AArch64 (D172) the scalar integer floor is 25.4 cycles per sample (one `smaddl` per multiply-add on one pipe), while GCC vectorizes CMSIS-DSP (16.40). Plan: C2 an x86-64 group size that fits the registers (as the `a64` module of D172; 4 to 6 outputs, measured on both models), and the history outputs grouped as in `a64`; C3 a form that the compilers vectorize while the sums stay exact: `i16 x i16` products summed in pairs in `i32` with an offset that maps the pair range `[-2^31 + 2^16, 2^31]` into `u32` (`pmaddwd` on x86, `smull` / `sadalp` on AArch64) and widened into `i64`, the offsets taken off once per output; the exact `i64` sum is unchanged, so the bits are those of the contract. Cortex-M4 must keep its count (C2 is `cfg(target_arch = "x86_64")`; C3 only where it measures faster).
- **`stereo_matrix_fallback` (`sse2` 1.45 / 1.21, `sse4.2` 1.42 / 1.06, `avx2` 1.06 / 1.61).** LLVM vectorizes the interleaved `(s, d)` loop with a deinterleave and a re-interleave (`shufps` pairs at `sse2`; at `avx2` 8 `vshufps` / `vpermpd` / `vunpck` / `vperm2f128` per 8 pairs, port 5 bound on `alderlake`), while CMSIS-DSP adds planar arrays. The work per pair is three additions in both. A form without deinterleave exists: `[s0, s0, s1, s1] + [d2, -d2, ...]` (`s - d2` equals `s + (-d2)` bit for bit); the portable forms tried here (lane arrays of 4 and 8) were turned back into gathers by LLVM, as the earlier try of I24-3. Plan: C4 more portable forms (a `u64` view of the pair, a sign mask applied by `xor` on the bits of `d2`); if none compiles to the in-lane form, an open item for the maintainer: an x86 form with `core::arch` shuffles inside the portable crate (the question of I24-3).
- **`stereo_pll_fallback` (`sse2` 1.24 / 1.22, `sse4.2` 1.23 / 1.22; ours 57.02 dB against 9.46 dB).** An accuracy trade (D2760) needs at most 1.20 on both models; the loop is latency bound (efficiency 0.19). Plan: C5 shorten the loop-carried chain without changing a bit (the lock detector and the integrator limit off the phase chain, independent work hoisted); target 1.20 or below on both models (about 3 % on `skylake`, 2 % on `znver2`).
- **`deemphasis_fallback` (`sse2`, `sse4.2` 1.15 on `skylake` only).** A recursion `y += alpha (x - y)` (contract order, no FMA) plus the denormal flush: the chain `subss`, `mulss`, `addss` (12 cycles on `skylake`) and the select of the short form (`and` / `andn` / `or` without `blendvps`). GNU Radio's chain is one product and one sum in `double` (8). Plan: C6 a select with one operation on the chain (the flushed value merged by `or` into a cleared `kept`, or `blendvps` at `sse4.2`); the remaining gap is the contract's chain and goes into the open item.
- **`dot_cq15_fallback` (`sse2` 1.23 / 1.24, `sse4.2` 1.15 on `znver2`), `mul_real_q15_fallback` (`avx2` 1.25 on `alderlake`), `deinterleave_cq15_fallback` (`avx2` 1.14 on `znver3`), `mag_sq_fallback` and `mul_fallback` (`avx512` 1.21 and 1.47 on `znver4`), `fir_interpolate_real_fallback` and `fir_interpolate_short_fallback` (`neon` 1.22 and 1.13 on `cortex-a72`).** Not analysed yet; the first look of each step reads the loops of both sides. Known leads: `dot_cq15` the exact `i64` sum (the pair trick of C3 applies); `mag_sq` / `mul` at `avx512` the same as at `avx2` but slower on `znver4` (the 512-bit deinterleave shuffles of LLVM's vectorization of the fallback; the `avx2` builds simulate 0.29 (`mag_sq`) and 0.51 (`mul`) on `znver3`); the `f32` interpolators on `cortex-a72` the group of 16 accumulators.

## Order

1. B5 (`scramble` gather), B4 (`mul` form), C2 (x86 group sizes of the Q15 FIR engines), A1 (bit reversal): the clearest causes.
2. One CI run of the changed entries on every level they touch (`nightly.yml`, `measure_only`), then the next group while it runs.
3. C3, A2, B1 to B3, C4 to C6 and the unanalysed rows.
4. Item 8 (the run-time dispatch of the FIR entries) and item 9 (SIMD128 references) in parallel steps of the same workflow.
5. Open items IO1-n for every part that stops (a contract question or work the library does not do), with numbers and options; statuses in `benches/limits.json` only from CI measurements.

## Status

- 2026-11-24: planned; target table and causes above.
- 2026-11-24: group A done (D2810, D2811): the four FFT fallbacks at `sse2` and `sse4.2` are `release`, 0.54 to 0.99 of CMSIS-DSP on both models (CI run 45479384022); the x86 stage pairs run on two samples per vector where no infinity or NaN can arise, bit for bit the scalar stages. `channelize_fallback` 4.5 to 7.6 % slower on x86 (information, still at most 0.38 of liquid-dsp). No open item.
