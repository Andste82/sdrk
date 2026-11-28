# O3a — Cycle optimisation round 3a: SIMD paths slower than their fallback, gaps without a reason

> Plan: [M13.1](M13.1-cycles-only.md) (cycles per sample, the like-with-like statuses of D2764 to D2779) · Previous round: [O2](O2-cycle-optimisation.md) (with O2b and O2c) · Open items it works on: IO2-28, IO2-29, IO1-7 ([open-items.md](../open-items.md)), issue #203 (the kernel gap inventory) · Next: [BACKLOG](BACKLOG.md)

> **Status: STATUS_LINE**

## Goal

The maintainer's order of 2026-11-28 ("#209 mergen, dann O3a starten (vergiss den rest nicht!)"), with the scope he approved as "prio 1: optimisations that are necessary (red, SIMD slower than its own fallback)" plus item 14 of the O2c list, on the branch `opt-round-3a` from `main` d05a65e6 (PR #209 merged):

1. **SIMD slower than its own fallback (IO2-28):** `fir_decimate_cq15` and `fir_interpolate_cq15` at `sse2` (1.22 / 1.23 and 1.12 / 1.15 of the `pmaddwd` fallbacks of D3017) and at `sse4.2` (about 1.00): at most 0.80 of the fallback on both models, bit-identical, the `cycle_waiver`s removed.
2. **Red without a reason** (`dev` above 1.10 of the library with only a waiver): `dot_cq15_fallback` at `sse2` and `sse4.2`, `mag_sq_fallback` at `avx512`, `fir_interpolate_real_fallback` and `fir_interpolate_short_fallback` at `neon`, `deinterleave_cq15_fallback` at `avx2`: at most 1.10 on both models, then `release`.
3. **`simd128` fallbacks (IO2-29, instructions per sample):** `fir_cq15_fallback`, `fir_interpolate_short_fallback`, `fir_interpolate_real_fallback`, `fft_4096_fallback`, `scale_fallback` (`stereo_pll_fallback` is more accurate already): at most 1.10 of the judged opponent (D2953).
4. **Item 14:** `fft_cq15_1024_fallback` judged against CMSIS-DSP's `arm_cfft_q15` (it was `info`), with `slower_because` of the kind `more accurate` where it is above 1.10, and then faster where possible (an accuracy trade needs at most 1.20 on both models).

What cannot reach its target gets an honest text in `benches/limits.json` and an open item.

## Rules

As in [O2](O2-cycle-optimisation.md#rules): speed is cycles per sample (`llvm-mca` with the corrections of D942) on the reference model and the counter-model of the level, instructions only at `simd128` and on Cortex-M4; the status only against a foreign library, like with like per level, our own code information; accuracy first (at most 1 dB below the library; numeric contracts unchanged); a fallback change must not be slower on the AArch64 scalar build nor raise the Cortex-M4 instructions per sample; `sdrk-core` keeps `deny(unsafe_code)` except the modules that `cargo xtask check heap` names; an explicit SIMD path at about 1.00 of its fallback is accepted (D3019). Measurements of record come from CI (`nightly.yml`, `measure_only`); locally only short checks (the reproduction of `costas` at `neon`, 86.13 / 54.10, passed with `/opt/llvm-mca-23.1.1-sdrk`). No test, limit or bound is loosened. Decision numbers D3100 to D3149, open items IO3-1 to IO3-n.

## Results

Measured in CI: runs 46149899815 (on 76d51055, `measure_only`: the `CQ15` FIR family, `dot_cq15`, `mag_sq` at the x86-64 levels), 46152296739 (on 3aa9f3ea, `measure_only`: `fir`, `fir_decimate`, `fir_interpolate`, `resample`, `deinterleave` at every level with a core model) and the full run 46157162937 (on the branch head of that time: every entry and level, 1836 records merged, 0 failed in `cargo xtask check cycles`; the counts of `count_wasm32` and `count_m4_qemu`). Ratios are reference model / counter model; models `sse2`, `sse4.2` `skylake` / `znver2`, `avx2` `znver3` / `alderlake`, `avx512` `znver4` / `sapphirerapids`, `neon` `cortex-a72` / `cortex-a76`.

### Item 1: SIMD paths slower than their own fallback (IO2-28, D3100)

The `CQ15` progressions paired the two components of each pair of samples with a shuffle (`pshuflw` + `pshufhw` at SSE2) and multiplied split taps (two `pmaddwd` per vector): the same inner loop as the `pmaddwd` fallback of D3017 (the compiled loops of both are instruction for instruction the same), plus a slower path for the outputs whose windows reach into the history (chunk by chunk through the history abstraction, 15 % of the instructions at `sse2`). The planar form (D3100) splits the samples of a batch into a plane of real and a plane of imaginary parts and multiplies them with full taps, one `pmaddwd` per eight products and no shuffle, when the taps keep every 32-bit lane exact (a sum of tap magnitudes of at most 65535 per lane; the usual filters, and the benchmark filters, are far below it); above it the split form runs. Bit-identical.

| Entry | Level | Ours (cycles per sample, ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback | Was | Waiver |
| --- | --- | --- | --- | --- | --- | --- |
| `fir_decimate_cq15` | `sse2` | 6.95 / 6.99 | 8.85 / 9.23 | 0.786 / 0.758 | 1.220 / 1.228 | removed |
| `fir_decimate_cq15` | `sse4.2` | 4.98 / 5.60 | 8.84 / 9.21 | 0.563 / 0.608 | 0.999 / 1.010 | removed |
| `fir_interpolate_cq15` | `sse2` | 38.19 / 42.35 | 59.70 / 60.72 | 0.640 / 0.697 | 1.116 / 1.153 | removed |
| `fir_interpolate_cq15` | `sse4.2` | 29.09 / 32.70 | 51.42 / 53.93 | 0.566 / 0.606 | 1.002 / 0.992 | removed |

The same form makes the other `CQ15` FIR entries faster (ours, cycles per sample, was → now): `fir_cq15` `sse2` 62.86 / 65.40 → 37.86 / 41.96, `avx2` 20.87 / 26.17 → 15.70 / 22.60, `avx512` 25.43 / 18.14 → 15.78 / 18.35; `fir_decimate_cq15` `avx2` 3.65 / 4.49 → 2.42 / 3.17, `avx512` 4.44 / 2.94 → 2.01 / 2.38; `fir_interpolate_cq15` `avx2` 23.29 / 28.27 → 15.92 / 22.82; `resample_cq15` `sse2` 65.33 / 68.39 → 40.31 / 43.57. `avx2-runtime` runs the same code at 1.04 to 1.06 of the fixed `avx2` build (information).

### Item 2: gaps above 1.10 of the library without a reason (IO1-7)

| Entry | Level | Change | Ours (cycles per sample, ref. / counter) | Library (cycles per sample, ref. / counter) | Ours / library | Was | Status | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dot_cq15_fallback` | `sse2` | `pmaddwd`, the taps split per block | 1.05 / 0.90 | 1.78 / 2.03 (CMSIS-DSP) | 0.588 / 0.440 | 1.232 / 1.244 | `release` (was `dev`) | D3101 |
| `dot_cq15_fallback` | `sse4.2` | the same | 1.05 / 0.89 | 2.04 / 1.49 | 0.515 / 0.598 | 0.995 / 1.153 | `release` (was `dev`) | D3101 |
| `mag_sq_fallback` | `avx512` | the squares first, two `vpermt2ps` | 0.29 / 0.16 | 0.31 / 0.27 (CMSIS-DSP) | 0.935 / 0.591 | 1.210 / 0.942 | `release` (was `dev`) | D3102 |
| `deinterleave_cq15_fallback` | `avx2` | blends, shifts and packs | 0.20 / 0.26 | 0.20 / 0.27 (VOLK `generic`) | 0.989 / 0.986 | 1.135 / 0.860 | `release` (was `dev`) | D3103 |
| `fir_interpolate_short_fallback` | `neon` | two columns per pass | 22.91 / 15.55 | 27.28 / 29.90 (CMSIS-DSP) | 0.840 / 0.520 | 1.133 / 0.789 | `release` (was `dev`) | D3105 |
| `fir_interpolate_real_fallback` | `neon` | the head outputs in groups, groups of 32 | 61.15 / 39.12 | 67.65 / 76.22 (CMSIS-DSP) | 0.904 / 0.513 | 1.224 / 0.819 | `release` (was `dev`) | D3104, D3108 |

`deinterleave_cq15_fallback` at `avx2` is 13 % faster on `znver3` and 15 % slower on `alderlake` than before (0.23 / 0.23), within 1.10 of VOLK on both. The head groups of D3104 also make the other fallbacks with windows one sample apart faster (`fir_fallback`, `fir_interpolate_fallback`, `fir_cq15_fallback`, `fir_interpolate_cq15_fallback` and the `f32` interpolator at every level, 2 % to 32 %); the progressions with windows more than one sample apart (`fir_decimate`, `resample`) keep the code of `main`: with the copy, `resample_real_fallback` at `avx512` was 0.3 % slower on `sapphirerapids` (1.1003 of liquid-dsp, CI run 46155369153) and `fir_decimate` 2.8 % more instructions on Cortex-M4 (CI run 46157162937), so the copy is a compile-time choice of the instantiations with windows one sample apart.

### Item 3: `simd128` (IO2-29, instructions per sample)

Judged opponent (D2953), count of `count_wasm32` of CI run 46157162937:

| Entry | Change | Ours (instructions per sample) | Opponent (instructions per sample) | Ours / opponent | Was | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `scale_fallback` | blocks of 32 floats (D3106) | 5.59 | 5.40 (CMSIS-DSP `arm_scale_f32`) | 1.03 | 1.14 | `release` (was `dev`) |
| `fir_interpolate_real_fallback` | the head outputs in groups (D3104) | 292.07 | 296.50 (CMSIS-DSP) | 0.99 | 1.18 | `release` (was `dev`) |
| `fir_interpolate_short_fallback` | two columns per pass (D3105) | 134.51 | 127.10 (CMSIS-DSP) | 1.06 | 1.38 | `release` (was `dev`) |
| `fir_cq15_fallback` | the head outputs in groups (D3104) | 897.87 | 785.69 (CMSIS-DSP `arm_fir_q15`) | 1.14 | 1.19 | `dev`, waiver, IO2-29 |
| `fft_4096_fallback` | none | 244.93 | 221.28 (CMSIS-DSP `arm_cfft_f32`) | 1.11 | 1.11 | `dev`, waiver, IO2-29 |
| `stereo_pll_fallback` | none | 189.06 | 143.86 (liquid-dsp) | 1.31 | 1.31 | `dev`, `slower_because` (`more accurate`) |

The planar progressions of D3100 at `simd128` (information against our fallback): `fir_cq15` 479.46 → 321.72, `fir_decimate_cq15` 93.03 → 58.53, `fir_interpolate_cq15` 504.01 → 326.14, `resample_cq15` 496.32 → 333.51 instructions per sample.

Cortex-M4 (instructions per sample, `count_m4_qemu`): `fir` 510.45 → 484.41 (the head groups); `fir_decimate` 112.28 → 115.44 in CI run 46157162937 with the copy present in the instantiations with windows more than one sample apart, which since 2a191736 keep the code of `main` (the final run below).

### Item 4: `fft_cq15_1024_fallback` against CMSIS-DSP (D3107, IO3-1)

Judged against CMSIS-DSP's `arm_cfft_q15` (it was `info`); 68.03 dB against 41.76 dB (block floating point, a rounding in every butterfly). No code change: the scalar stages were not made faster in O3a.

| Level | Ours (cycles per sample, ref. / counter) | CMSIS-DSP (cycles per sample, ref. / counter) | Ours / CMSIS-DSP | Status | Text |
| --- | --- | --- | --- | --- | --- |
| `sse2` | 83.13 / 91.71 | 44.75 / 56.34 | 1.858 / 1.628 | `dev` (was `info`) | waiver, above 1.50, awaiting the maintainer (IO3-1) |
| `sse4.2` | 76.35 / 77.17 | 44.80 / 56.34 | 1.704 / 1.370 | `dev` (was `info`) | waiver, above 1.50, awaiting the maintainer (IO3-1) |
| `avx2` | 46.25 / 51.39 | 37.69 / 38.63 | 1.227 / 1.330 | `dev` (was `info`) | `slower_because`, `more accurate` |
| `avx512` | 42.83 / 50.08 | 37.40 / 39.49 | 1.145 / 1.268 | `dev` (was `info`) | `slower_because`, `more accurate` |
| `neon` | 120.91 / 100.37 | 64.42 / 52.33 | 1.877 / 1.918 | `dev` (was `info`) | waiver, above 1.50, awaiting the maintainer (IO3-1) |
| `simd128` | 537.22 (instructions) | 378.19 | 1.42 | `dev` (was `info`) | `slower_because`, `more accurate` |

No level is within the 1.20 of an accuracy trade. Above 1.50 the gate admits a `dev` entry only with a waiver that awaits the maintainer (D2779), so the reason stands in the waiver there.

## Open items

- IO3-1: `fft_cq15_1024_fallback` above the library at every level, above 1.50 at `sse2`, `sse4.2` and `neon` (the maintainer's decision, or a faster scalar form later).
- IO3-2: a unit test of the `C32` FFT fallback fails at `x86-64-v3` on `main` as well (not a change of O3a).
- IO2-29 stays open for `fir_cq15_fallback` (1.14) and `fft_4096_fallback` (1.11) at `simd128`.
- Closed: I44-10 (`fir_interpolate_short_fallback` at `neon` and `simd128` `release`).
- Closed: IO1-7 (every row `release`), IO2-28 part (a); IO2-28 part (b) stays (accepted by D3019).

