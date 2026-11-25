# O2 — Cycle optimisation round 2: SIMD paths that bring nothing, same-work references, accuracy

> Plan: [M13.1](M13.1-cycles-only.md) (cycles per sample, the like-with-like statuses of D2764 to D2779) · Previous round: [O1](O1-cycle-optimisation.md) · Open items it works on: IO1-1 to IO1-9, I13.1-9, I13.1-12, I13.1-13, I13.1-14 ([open-items.md](../open-items.md)), issue #203 (the kernel gap inventory) · Next: [BACKLOG](BACKLOG.md)

> **Status: planned** (2026-11-25 on the maintainer's order of the same day, after O1 was merged; branch `opt-round-2` from `main` 05ff2d25). Decision numbers D2950 to D2999, open items IO2-1 to IO2-n. No seeds.

## Goal

Every SIMD entry faster than its own fallback, every entry at least as fast as its foreign reference library (or a stated reason why it does more work), and every entry at least as accurate as its library. The maintainer's order of 2026-11-25 ("starte o2 nachdem o1 gemergt wurde"), in seven groups:

- **Group 0 (first, read-only):** an inventory of every place where a SIMD path is bypassed or delegated to the fallback, each with its reason and date (below).
- **Group 1, same-work references.** The maintainer's rule: if no opponent does the same work, find another one, or state clearly why we are slower and what we do more. `am_mod_fallback` (IO1-3: liquid-dsp's `ampmodem` does no clamp, no NaN rule, no carrier product) and `convert_out_cu8` / `convert_out_real_ru8` (IO1-2: our NaN rule and the exact rounding): a library or a composition of library calls that does the same work (CMSIS-DSP `arm_clip_f32` + `arm_scale_f32` + `arm_offset_f32` + the carrier product for the scalar fallback; KFR `clamp` / `select` / `isnan` expressions for SIMD; with its NaN behaviour checked). If there is none: what we do more, why, what it costs (the ratio), what it gains (measured), and the status `info` with that reason.
- **Group 2, accuracy.** `oscillate` (133.2 dB against SLEEF 141.7 and liquid-dsp 140.8, I13.1-13) first, then `deemphasis_fallback`, `fir_decimate_fallback`, `am_mod_fallback` at `neon` where still more than 1 dB below their library: at least as accurate without losing speed.
- **Group 3a, SIMD paths without benefit, FIR family.** The `Q15` / `CQ15` FIR SIMD branches behind the switch of D65 (`runs_fallback` in `sdrk-backend-simd/src/fir.rs`) compute one output at a time with a horizontal reduction: rewrite them in the grouped form (several outputs per vector, no horizontal reduction per output) so that each level beats the fallback (target at most 0.80 of the fallback on both models), then remove the rows of the switch. The `Q15` kernels at `sse2` without a SIMD branch (`am_mod_cq15`, `biquad_cq15`, `biquad_real_q15`, `mul_cq15`, `mul_real_q15`, `scale_cq15`, `resample_real_q15`, ...): the missing SSSE3 / SSE4.1 instructions (`pmulhrsw`, `pmulld`) replaced by SSE2 sequences, bit-identical.
- **Group 3b.** `am_mod` gets a SIMD branch. Single cases where the SIMD entry is no faster than its fallback: `mag` at `avx512` and `neon`, `mul` at `sse2`, `stereo_matrix` and `stereo_matrix_cq15` at `neon`, `dot_cq15` at `neon`, `interleave` / `interleave_cq15` / `deinterleave` / `convert_cq15_cs8` at `avx2` (a real SIMD gain, or the duplicate deleted with the statement that the compiler vectorises the fallback as well). `equalize` (all five entries): its inner dot product and tap update vectorised (not the loop itself).
- **Group 4, above 1.10 of the library.** `fft_cq15_1024` and `fft_cq15_1024_full_scale` at `neon` against Ne10 (I13.1-12); the 8-bit `convert_out` fallbacks against VOLK `generic` (I13.1-14); IO1-1 (if still open), IO1-4 (`stereo_matrix_fallback`, with x86 shuffles allowed, D2952), IO1-5 (`fir_decimate_cq15_fallback` x86), IO1-6 (the `Q15` interpolator fallbacks on `znver2`), IO1-7 (`dot_cq15_fallback`, `mul_real_q15_fallback`, `deinterleave_cq15_fallback`, `mag_sq_fallback` at `avx512`), IO1-8, IO1-9 (`mul_fallback` at `avx512`, D2951); `channelize_fallback` (4.5 to 7.6 % slower on x86 since O1's FFT change, D2811 / D2841); `fir_interpolate_real_q15` at `avx2-runtime` (1.023 / 1.071 of the fixed `avx2` build, information).
- **Group 5, checks (report only, no failure).** `cargo xtask check cycles` reports every SIMD entry that needs more than 0.80 of its fallback's cycles on a model, and every kernel entry-level without a foreign reference (the inventory of issue #203).

Not in O2: the control loops (`costas*`, `symbol_sync*`, `agc` at `neon`), reserved for a later round O3; the entries of issue #203 without a reference (low priority), except where a reference falls out of group 1.

At the end: the kernel gap inventory of issue #203 re-run on the new `main` and the issue updated in its structure (A: no foreign reference, B: SIMD no faster than the fallback), with the changes since its creation.

## Rules

- **Speed** is cycles per sample (`llvm-mca` with the corrections of D942) on the reference model and the counter-model of the level: `sse2`, `sse4.2` `skylake` / `znver2`; `avx2`, `avx2-runtime` `znver3` / `alderlake`; `avx512` `znver4` / `sapphirerapids`; `neon` `cortex-a72` / `cortex-a76`. Instructions only at `simd128` and on Cortex-M4. A branch costs its cycles plus mispredictions times the penalty of the core (D1611, D1612).
- **Judged against a foreign library**, like with like per level (`cargo xtask check` enforces it); our own code is information only. `release`: at most 1.10 of the library on both models; `dev`: at most 1.50, with an open item; above 1.50 `dev` with an open item that this round must solve (D2950). At `simd128` the judged opponent is the stronger of KFR's wasm32 build and VOLK's generic C compiled with `-O3 -msimd128` (D2953).
- **Accuracy first (D34):** at most 1 dB below the library; an accuracy trade only at least 3 dB better, library below 100 dB, at most 1.20 on both models. Numeric contracts (bit-exactness, chunk invariance, no FMA where the contract says so) do not change without the maintainer: such a part stops with an open item.
- **Fallbacks prefer Arm:** a fallback change must not be slower on the AArch64 scalar build, and the Cortex-M4 instructions per sample (`cargo xtask count --platform cortex-m4`) must not rise; auto-vectorisation gains are welcome.
- No test, limit or bound is loosened, skipped or commented out. Committed numbers come from the CI container only (`nightly.yml`); locally only short checks of a few entries.

## Decisions of the maintainer (2026-11-25)

Recorded in [decisions-log.md](../decisions-log.md):

- **D2950** (decision 2): an entry above 1.50 of its library is `dev` with an open item and must be solved in this round, or, where the library provably does less work, handled as in group 1.
- **D2951** (decision 3, IO1-9): the f64 `Reference` implementation of the `C32` product gets a fixed order of its sums, such that the faster AVX-512 grouped product fallback (0.999 / 0.682 of CMSIS-DSP) gives the same bits, the sign of a NaN included; that faster form is taken.
- **D2952** (decision 4, IO1-4): x86 shuffle intrinsics (`core::arch`) are allowed in the portable fallback of `stereo_matrix`, gated narrowly to x86-64 and tested bit for bit against the portable form.
- **D2953** (decision 5): at `simd128` the judged opponent of an entry is the stronger (fewer instructions per sample) of KFR's wasm32 build and VOLK's generic C compiled with `-O3 -msimd128`; the other is information.

## Group 0: where a SIMD path is bypassed or delegated to the fallback

Read-only inventory of `main` 05ff2d25 (`crates/facade/sdrk/src/delegations.rs`, `crates/backends/sdrk-backend-simd/src`). "Fallback" is `sdrk_core::Fallback`, compiled for the level of the build. The date is the commit that introduced the switch (`git blame`). The rows marked with a group are work of this round.

### 0.1 Kernels the facade never gives to the SIMD backend

| Kernel / type | Where | Reason | Decision | Date | O2 |
| --- | --- | --- | --- | --- | --- |
| `deemphasis` `Q15` | `delegations.rs` (`=> Fallback`) | one-pole recursion, no SIMD path written (M06); "not accelerated" in `docs/backends.md` | none recorded | 2026-10-18 | |
| `dc_block` `CQ15` | `delegations.rs` | as `deemphasis` `Q15` | none recorded | 2026-10-18 | |
| `dc_block_real` `Q15` | `delegations.rs` | as `deemphasis` `Q15` | none recorded | 2026-10-25 | |
| `agc` `CQ15` | `delegations.rs` | per-sample gain feedback (M20) | M20 | 2026-10-25 | O3 |
| `squelch` `C32`, `CQ15` | `delegations.rs` | first-order power recursion bounds the loop; the gate decision depends on it (M19) | M19 | 2026-10-25 | |
| `noise_blank` `C32`, `CQ15`; `auto_notch` `f32` | `delegations.rs` | per-sample recursions (M33, specification test 5) | M33 | 2026-11-05 | |
| `preemph_limit` `f32`, `Q15` | `delegations.rs` | pre-emphasis and envelope recursions (M25) | D183 | 2026-11-01 | |
| `stereo_pll` `f32`, `Q15` | `delegations.rs` | phase of every sample depends on the previous detector output (M24) | D110 | 2026-11-01 | |
| `preemph_limit_stereo`, `stereo_encode` `C32` | `delegations.rs` (by hand) | recursion; a table lookup per sample (M36) | M36 | 2026-11-05 | |
| `rds_shape` `f32`, `stereo_encode_rds` `C32` | `delegations.rs` (by hand) | four interpolated lookups per sample at an f64 position; a gather per sample (M42) | D757 | 2026-11-06 | |
| `symbol_sync` `C32`, `f32` | `delegations.rs` | recursion over the strobes (M38) | M38 | 2026-11-05 | O3 |
| `costas` `C32` | `delegations.rs` (by hand) | recursion over the symbols (M40) | D608 | 2026-11-05 | O3 |
| `equalize` `C32` | `delegations.rs` (by hand); `equalize.rs` (`Simd` runs the fallback's loop) | a dispatched SIMD `dot_complex` per symbol on 11 taps cost more than the fallback's own filter part, which the compiler vectorises | D841 | 2026-11-07 | 3b |

### 0.2 Whole-call switches by level or target

| Kernel / type | Where | Levels | What runs | Reason (numbers as recorded) | Decision | Date | O2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fir`, `fir_decimate`, `fir_interpolate`, `resample` `CQ15`; `fir_interpolate_real`, `resample_real` `Q15` | `fir.rs` `runs_fallback` | `sse2`: all six | grouped fallback engine (D59) | one output at a time with a horizontal reduction; SIMD / fallback 1.87 to 2.28 on `skylake`, 1.21 to 1.45 on `znver2` | D65 | 2026-10-31 | 3a |
| same | `fir.rs` `runs_fallback` | `avx2`: all but `resample_real` | grouped fallback engine | SIMD / fallback 1.16 to 1.66 (`znver3`), 1.14 to 1.64 (`alderlake`) | D65 | 2026-10-31 | 3a |
| same | `fir.rs` `runs_fallback` | `avx512`: `fir`, `fir_interpolate`, `fir_interpolate_real` | grouped fallback engine | 1.15 to 1.33 / 1.09 to 1.32 | D65 | 2026-10-31 | 3a |
| same | `fir.rs` `runs_fallback` | `neon`: `fir_decimate`, `fir_interpolate_real` | fallback (exact lanes of D2830; scalar ring loop) | 1.76 / 1.63; 1.35 / 1.08 | D65, D2830 | 2026-11-25 | 3a |
| same | `fir.rs` `runs_fallback` | `simd128`: `fir_interpolate`, `fir_interpolate_real` | fallback (scalar ring loop) | 1.54 and 1.03 of the fallback's instructions | D65 | 2026-11-01 | 3a |
| `fir` `CQ15`, `fir_interpolate` `CQ15`, `fir_interpolate_real` `Q15` | `fir.rs` `runtime_avx2` | `avx2-runtime` (a build that detects exactly AVX2) | the fallback compiled for `x86-64-v3` (`FallbackAt`) | the row `avx2` of `runs_fallback`; the build's own fallback would be SSE2 code | D2840 | 2026-11-25 | 3a |
| `biquad_real` `Q15` | `biquad.rs` (`sse2_only`) | `sse2` | fallback | no `pmuldq` / `pminsd` (SSE4.1); the `f64` path 33.7 / 23.0 against 15.1 / 15.1 cycles | the integer-lane biquads | 2026-10-28 | 3a |
| `biquad` `CQ15` | `biquad.rs` (`sse2_only`) | `sse2` | fallback | the `f64` path bound by its recursion latency: 32.45 / 20.44 against 24.43 / 31.16 | D341 | 2026-11-04 | 3a |
| `agc` `C32` | `agc.rs` (`cfg!(target_arch = "aarch64")`) | `neon` | fallback | the plain loop is faster: 39.90 / 21.76 against 47.07 / 22.46 cycles | D335 | 2026-11-04 | O3 |
| `dc_block` `C32` | `recursion.rs` (`cfg!(target_family = "wasm")`) | `simd128` | fallback | 36.56 against 26.53 instructions per sample | the f64 block scan of the recursions | 2026-10-29 | |
| `add` `CQ15` | `math.rs` (`cfg!(target_family = "wasm")`) | `simd128` | fallback (groups of eight, D318) | 3.61 against 2.33 instructions per sample | D318, D322 | 2026-11-04 | |
| `add_real` `Q15` | `math.rs` (`cfg!(target_family = "wasm")`) | `simd128` | fallback (groups of eight, D294) | 1.83 against 1.19 instructions per sample | D294, D322 | 2026-11-04 | |

### 0.3 Levels without a path whose caller runs the fallback

| Kernel / type | Where | Levels | What runs | Reason | Decision | Date | O2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mul_real` `Q15` | `math.rs` `mul_real_q15` | `sse2` | fallback, whole call | no rounding high multiplication (`pmulhrsw`, SSSE3) | D332 | 2026-11-04 | 3a |
| `mul` `CQ15` | `math.rs` `mul_cq15` | `sse2`, `simd128` | fallback, whole call | no path written for these levels (`docs/backends.md`) | D332 | 2026-11-04 | 3a |
| `scale` `CQ15` | `scale.rs` `scale_q15_level` | `sse2` | fallback | no rounding high multiplication | M13 | 2026-10-29 | 3a |
| `am_mod` `CQ15` | `am_mod.rs` `am_cq15_level` | `sse2` | the `i32` loop of the fallback at the level of the dispatch | no rounding high multiplication | M35 | 2026-11-05 | 3a |
| `am_mod` `C32` | `am_mod.rs` | every level | the fallback's loop compiled at the level of the dispatch (no hand-written branch) | LLVM vectorises the loop like the fallback | M35 | 2026-11-05 | 3b |
| `dot` `CQ15` | `dot.rs` `dot_cq15_level` | `neon` | fallback, whole call | the compiled fallback (`ld2`, `smlal`) 0.90 against 0.95 / 1.04 cycles of the `uzp`/`smull`/`sadalp` path | M13 | 2026-10-29 | 3b |
| `stereo_matrix` `CQ15` | `stereo.rs` `matrix_q15` | `simd128` | fallback | the `i32x4.dot_i16x8_s` branch took 5.90 against 5.40 instructions (the module comment cites D156; the decision is D158) | D158 | 2026-11-01 | |
| `stereo_matrix` `CQ15` | `stereo.rs` | `neon` | its NEON branch, kept although it ties the fallback (1.00) | not worse (D148) | D159 | 2026-11-01 | 3b |

### 0.4 Calls that leave the SIMD path by parameter or data (every level)

Not level switches; they keep the result of the contract for inputs the vector code does not cover. Listed for completeness, no O2 work.

| Kernel | Where | Condition that runs the fallback |
| --- | --- | --- |
| `convert`, `convert_real`, `convert_out`, `convert_out_real` | `convert.rs`, `convert_out.rs` | a format other than `U8`, `I8`, little-endian `I16` (full container, power-of-two scale, no swap); lengths that do not fit; the odd last value of `convert_real` `Q15` |
| `fir` `CQ15`, `fir_decimate` `CQ15` | `fir.rs` | more than `PREP_TAPS` = 256 taps |
| `fir_interpolate`, `resample` `CQ15`; `Q15` real forms | `fir.rs` | prepared taps of all branches above `CAPACITY` = 384 samples |
| `halfband_decimate`, `halfband_decimate_real`, `real_to_complex` | `halfband.rs` | more than 1023 taps, asymmetric `C32` taps, a `CQ15` tap sum out of range |
| `dc_block`, `dc_block_real`, `deemphasis` `f32` / `C32` | `recursion.rs` | a NaN or infinity in the state or input, a pole below 1e-8, `alpha = 1` |
| `am_mod` | `am_mod.rs` | carrier or depth outside the valid range |
| `fm_mod` | `modulate.rs` | `C32`: no input limit for the scale (steps beyond 2^51); `CQ15`: a step that is not exact in `k / 2^15`; the `C32` scalar runs of `SCALAR_RUN` after each vector run |
| `oscillate` | `math.rs` | a harmonic outside 1 to 8; the rest after the last whole vector |
| `dot` `CQ15`, `dot_real` `Q15` | `dot.rs` | no partial result from the level (see 0.3) |
| `scale`, `stereo_matrix`, `qam_soft_bits`, `scramble` | `scale.rs`, `stereo.rs`, `qam.rs`, `scramble.rs` | the rest of a call shorter than one vector step |

## Targets

From the full CI measurement of every entry on every level on 05ff2d25 (CI run 45641571494, merged into `benches/results/cycles/*.json`; the `simd128` counts of its job `count_wasm32`, unchanged against the committed file) and `benches/limits.json`. Cycles per sample on the reference model / the counter-model of the level; a ratio is ours divided by the other on the same model; "own fallback" is the `_fallback` entry of the same level (a `_fallback` entry has none; `avx2-runtime` is judged through `avx2`, D2764). SNR 999 is an exact result. Status from `benches/limits.json`.

### Group 1: same-work references

| Entry | Level | Ours (cycles per sample, ref. / counter) | Library (cycles per sample, ref. / counter) | Ours / library (ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback (ref. / counter) | Models (ref. / counter) | SNR in dB (ours / library) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `am_mod_fallback` | `sse2` | 1.09 / 1.16 | 0.59 / 0.72 | 1.842 / 1.613 | — / — | — / — | skylake / znver2 | 150.80 / 150.80 | `dev` |
| `am_mod_fallback` | `sse4.2` | 1.09 / 1.16 | 0.59 / 0.72 | 1.842 / 1.613 | — / — | — / — | skylake / znver2 | 150.80 / 150.80 | `dev` |
| `am_mod_fallback` | `avx2` | 0.65 / 0.47 | 0.53 / 0.54 | 1.225 / 0.867 | — / — | — / — | znver3 / alderlake | 150.80 / 150.80 | `dev` |
| `am_mod_fallback` | `avx512` | 0.38 / 0.30 | 0.53 / 0.54 | 0.726 / 0.560 | — / — | — / — | znver4 / sapphirerapids | 150.80 / 150.80 | `release` |
| `am_mod_fallback` | `neon` | 2.91 / 1.52 | 1.28 / 1.03 | 2.273 / 1.478 | — / — | — / — | cortex-a72 / cortex-a76 | 150.80 / 152.07 | `dev` |
| `am_mod` | `sse2` | 1.09 / 1.16 | — / — | — / — | 1.09 / 1.16 | 1.000 / 1.000 | skylake / znver2 | 150.80 / — | `info` |
| `am_mod` | `sse4.2` | 1.09 / 1.16 | — / — | — / — | 1.09 / 1.16 | 1.000 / 1.000 | skylake / znver2 | 150.80 / — | `info` |
| `am_mod` | `avx2` | 0.65 / 0.47 | — / — | — / — | 0.65 / 0.47 | 1.000 / 1.000 | znver3 / alderlake | 150.80 / — | `info` |
| `am_mod` | `avx2-runtime` | 0.65 / 0.47 | — / — | — / — | — / — | — / — | znver3 / alderlake | 150.80 / — | `info` |
| `am_mod` | `avx512` | 0.38 / 0.30 | — / — | — / — | 0.38 / 0.30 | 1.000 / 1.000 | znver4 / sapphirerapids | 150.80 / — | `info` |
| `am_mod` | `neon` | 2.91 / 1.52 | — / — | — / — | 2.91 / 1.52 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 150.80 / — | `info` |
| `convert_out_cu8` | `sse2` | 1.59 / 1.55 | 1.27 / 1.27 | 1.252 / 1.225 | 5.43 / 5.74 | 0.292 / 0.271 | skylake / znver2 | 47.27 / 47.27 | `dev` |
| `convert_out_cu8` | `sse4.2` | 1.57 / 1.52 | 1.33 / 1.39 | 1.177 / 1.094 | 4.70 / 4.41 | 0.334 / 0.346 | skylake / znver2 | 47.27 / 47.27 | `dev` |
| `convert_out_cu8` | `avx2` | 0.53 / 0.67 | 0.54 / 0.60 | 0.985 / 1.114 | 1.67 / 1.82 | 0.319 / 0.369 | znver3 / alderlake | 47.27 / 47.27 | `dev` |
| `convert_out_cu8` | `avx2-runtime` | 0.53 / 0.66 | — / — | — / — | — / — | — / — | znver3 / alderlake | 47.27 / — | `dev` |
| `convert_out_cu8` | `avx512` | 0.53 / 0.49 | 10.20 / 8.39 | 0.052 / 0.058 | 1.22 / 1.33 | 0.435 / 0.370 | znver4 / sapphirerapids | 47.27 / 47.27 | `release` |
| `convert_out_cu8` | `neon` | 2.91 / 2.09 | 2.66 / 2.15 | 1.094 / 0.974 | 7.14 / 4.35 | 0.408 / 0.481 | cortex-a72 / cortex-a76 | 47.27 / 47.27 | `release` |
| `convert_out_cu8_fallback` | `sse2` | 5.43 / 5.74 | 12.29 / 14.25 | 0.442 / 0.402 | — / — | — / — | skylake / znver2 | 47.27 / 47.27 | `release` |
| `convert_out_cu8_fallback` | `sse4.2` | 4.70 / 4.41 | 2.83 / 2.72 | 1.658 / 1.621 | — / — | — / — | skylake / znver2 | 47.27 / 47.27 | `dev` |
| `convert_out_cu8_fallback` | `avx2` | 1.67 / 1.82 | 1.22 / 1.49 | 1.371 / 1.224 | — / — | — / — | znver3 / alderlake | 47.27 / 47.27 | `dev` |
| `convert_out_cu8_fallback` | `avx512` | 1.22 / 1.33 | 0.68 / 1.27 | 1.805 / 1.041 | — / — | — / — | znver4 / sapphirerapids | 47.27 / 47.27 | `dev` |
| `convert_out_cu8_fallback` | `neon` | 7.14 / 4.35 | 4.90 / 2.84 | 1.456 / 1.532 | — / — | — / — | cortex-a72 / cortex-a76 | 47.27 / 47.27 | `dev` |
| `convert_out_real_ru8` | `sse2` | 0.81 / 0.79 | 0.64 / 0.64 | 1.260 / 1.233 | 2.72 / 2.88 | 0.296 / 0.274 | skylake / znver2 | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8` | `sse4.2` | 0.80 / 0.77 | 0.67 / 0.70 | 1.190 / 1.105 | 2.36 / 2.21 | 0.338 / 0.350 | skylake / znver2 | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8` | `avx2` | 0.27 / 0.34 | 0.28 / 0.30 | 0.992 / 1.132 | 0.84 / 0.92 | 0.326 / 0.376 | znver3 / alderlake | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8` | `avx2-runtime` | 0.27 / 0.34 | — / — | — / — | — / — | — / — | znver3 / alderlake | 47.25 / — | `dev` |
| `convert_out_real_ru8` | `avx512` | 0.27 / 0.25 | 5.11 / 4.20 | 0.054 / 0.060 | 0.61 / 0.67 | 0.445 / 0.377 | znver4 / sapphirerapids | 47.25 / 47.25 | `release` |
| `convert_out_real_ru8` | `neon` | 1.47 / 1.06 | 1.34 / 1.08 | 1.102 / 0.985 | 3.58 / 2.18 | 0.411 / 0.487 | cortex-a72 / cortex-a76 | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8_fallback` | `sse2` | 2.72 / 2.88 | 6.15 / 7.13 | 0.443 / 0.404 | — / — | — / — | skylake / znver2 | 47.25 / 47.25 | `release` |
| `convert_out_real_ru8_fallback` | `sse4.2` | 2.36 / 2.21 | 1.42 / 1.36 | 1.661 / 1.623 | — / — | — / — | skylake / znver2 | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8_fallback` | `avx2` | 0.84 / 0.92 | 0.61 / 0.75 | 1.372 / 1.226 | — / — | — / — | znver3 / alderlake | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8_fallback` | `avx512` | 0.61 / 0.67 | 0.34 / 0.64 | 1.797 / 1.046 | — / — | — / — | znver4 / sapphirerapids | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8_fallback` | `neon` | 3.58 / 2.18 | 2.46 / 1.42 | 1.457 / 1.533 | — / — | — / — | cortex-a72 / cortex-a76 | 47.25 / 47.25 | `dev` |

`simd128` (instructions per sample):

| Entry | Ours (instructions per sample) | Judged library (instructions per sample) | VOLK `-O3 -msimd128` (instructions per sample) | Ours / judged | Ours / stronger (D2953) | Own fallback (instructions per sample) | Ours / own fallback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `am_mod_fallback` | 10.54 | 46.02 | — | 0.229 | 0.229 | — | — | `release` |
| `am_mod` | 10.54 | — | — | — | — | 10.54 | 1.000 | `info` |
| `convert_out_cu8` | 15.17 | 29.03 | — | 0.523 | 0.523 | 26.57 | 0.571 | `info` |
| `convert_out_cu8_fallback` | 26.57 | 27.03 | — | 0.983 | 0.983 | — | — | `release` |
| `convert_out_real_ru8` | 7.63 | 14.53 | — | 0.525 | 0.525 | 13.31 | 0.573 | `info` |
| `convert_out_real_ru8_fallback` | 13.31 | 13.53 | — | 0.984 | 0.984 | — | — | `release` |

Above 1.50 (D2950, to be solved in O2 or handled by group 1): `am_mod_fallback` at `sse2` and `sse4.2` (1.842 / 1.613) and `neon` (2.273 on `cortex-a72`). `avx2` 1.225 on `znver3`; `avx512` and `simd128` are `release`. The SIMD entry `am_mod` is the fallback's loop on every level (1.000, group 3b). `convert_out_cu8` and `convert_out_real_ru8` (SIMD) are `dev` at `sse2` (1.25 / 1.23), `sse4.2` (1.18 / 1.09 and 1.19 / 1.105) and `avx2` (1.114 and 1.132 on `alderlake`); `release` at `avx512`, `cu8` also at `neon` (`ru8` 1.102 on `cortex-a72`). Same SNR as VOLK (47.27 / 47.25 dB): the extra work (NaN rule, exact rounding) shows in no number of the measurement, so group 1 has to state what it buys. Their fallbacks are in group 4.

### Group 2: accuracy

| Entry | Level | Ours (cycles per sample, ref. / counter) | Library (cycles per sample, ref. / counter) | Ours / library (ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback (ref. / counter) | Models (ref. / counter) | SNR in dB (ours / library) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `oscillate` | `sse2` | 2.30 / 2.28 | 36.86 / 28.44 | 0.062 / 0.080 | 4.90 / 6.35 | 0.470 / 0.359 | skylake / znver2 | 133.23 / 141.71 | `release` |
| `oscillate` | `sse4.2` | 2.29 / 2.57 | 22.92 / 21.71 | 0.100 / 0.118 | 4.90 / 6.35 | 0.467 / 0.404 | skylake / znver2 | 133.23 / 141.71 | `release` |
| `oscillate` | `avx2` | 1.40 / 1.45 | 4.93 / 4.57 | 0.283 / 0.316 | 2.27 / 4.03 | 0.615 / 0.359 | znver3 / alderlake | 133.23 / 141.71 | `release` |
| `oscillate` | `avx2-runtime` | 1.40 / 1.45 | — / — | — / — | — / — | — / — | znver3 / alderlake | 133.23 / — | `release` |
| `oscillate` | `avx512` | 0.43 / 2.08 | 6.46 / 8.59 | 0.067 / 0.242 | 0.76 / 3.79 | 0.570 / 0.549 | znver4 / sapphirerapids | 133.23 / 141.71 | `release` |
| `oscillate` | `neon` | 5.55 / 3.28 | 35.40 / 23.70 | 0.157 / 0.138 | 8.10 / 10.06 | 0.685 / 0.326 | cortex-a72 / cortex-a76 | 133.23 / 141.71 | `release` |
| `oscillate_fallback` | `sse2` | 4.90 / 6.35 | 56.29 / 51.64 | 0.087 / 0.123 | — / — | — / — | skylake / znver2 | 133.23 / 140.77 | `release` |
| `oscillate_fallback` | `sse4.2` | 4.90 / 6.35 | 51.98 / 50.82 | 0.094 / 0.125 | — / — | — / — | skylake / znver2 | 133.23 / 140.77 | `release` |
| `oscillate_fallback` | `avx2` | 2.27 / 4.03 | 31.51 / 80.72 | 0.072 / 0.050 | — / — | — / — | znver3 / alderlake | 133.23 / 140.77 | `release` |
| `oscillate_fallback` | `avx512` | 0.76 / 3.79 | 78.52 / 90.69 | 0.010 / 0.042 | — / — | — / — | znver4 / sapphirerapids | 133.23 / 140.77 | `release` |
| `oscillate_fallback` | `neon` | 8.10 / 10.06 | 52.62 / 46.15 | 0.154 / 0.218 | — / — | — / — | cortex-a72 / cortex-a76 | 133.23 / 140.77 | `release` |
| `deemphasis_fallback` | `sse2` | 12.09 / 9.76 | 13.15 / 17.11 | 0.920 / 0.570 | — / — | — / — | skylake / znver2 | 145.36 / 151.98 | `release` |
| `deemphasis_fallback` | `sse4.2` | 12.09 / 9.76 | 13.15 / 17.11 | 0.920 / 0.570 | — / — | — / — | skylake / znver2 | 145.36 / 151.98 | `release` |
| `deemphasis_fallback` | `avx2` | 9.55 / 11.58 | 10.80 / 23.14 | 0.884 / 0.501 | — / — | — / — | znver3 / alderlake | 145.36 / 151.98 | `release` |
| `deemphasis_fallback` | `avx512` | 10.21 / 11.11 | 10.80 / 23.14 | 0.945 / 0.480 | — / — | — / — | znver4 / sapphirerapids | 145.36 / 151.98 | `release` |
| `deemphasis_fallback` | `neon` | 15.07 / 10.06 | 18.10 / 21.07 | 0.833 / 0.477 | — / — | — / — | cortex-a72 / cortex-a76 | 145.36 / 151.98 | `release` |
| `fir_decimate_fallback` | `sse2` | 15.09 / 15.82 | 38.80 / 23.74 | 0.389 / 0.666 | — / — | — / — | skylake / znver2 | 133.29 / 134.33 | `release` |
| `fir_decimate_fallback` | `sse4.2` | 15.09 / 15.82 | 38.81 / 23.69 | 0.389 / 0.668 | — / — | — / — | skylake / znver2 | 133.29 / 134.33 | `release` |
| `fir_decimate_fallback` | `avx2` | 11.63 / 11.54 | 71.28 / 32.88 | 0.163 / 0.351 | — / — | — / — | znver3 / alderlake | 133.29 / 134.33 | `release` |
| `fir_decimate_fallback` | `avx512` | 11.12 / 10.28 | 71.32 / 38.89 | 0.156 / 0.264 | — / — | — / — | znver4 / sapphirerapids | 133.29 / 134.33 | `release` |
| `fir_decimate_fallback` | `neon` | 32.76 / 25.71 | 34.10 / 41.23 | 0.961 / 0.624 | — / — | — / — | cortex-a72 / cortex-a76 | 133.29 / 134.33 | `release` |
| `am_mod_fallback` | `neon` | 2.91 / 1.52 | 1.28 / 1.03 | 2.273 / 1.478 | — / — | — / — | cortex-a72 / cortex-a76 | 150.80 / 152.07 | `dev` |

`simd128` (instructions per sample):

| Entry | Ours (instructions per sample) | Judged library (instructions per sample) | VOLK `-O3 -msimd128` (instructions per sample) | Ours / judged | Ours / stronger (D2953) | Own fallback (instructions per sample) | Ours / own fallback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `oscillate` | 22.04 | — | — | — | — | 28.28 | 0.779 | `info` |
| `oscillate_fallback` | 28.28 | 175.01 | — | 0.162 | 0.162 | — | — | `release` |
| `deemphasis_fallback` | 27.53 | 150.07 | — | 0.183 | 0.183 | — | — | `release` |
| `fir_decimate_fallback` | 147.84 | 138.12 | — | 1.070 | 1.070 | — | — | `release` |

Below the library by more than 1 dB: `oscillate` and `oscillate_fallback` 133.23 dB against SLEEF 141.71 dB (8.5 dB) and liquid-dsp 140.77 dB (7.5 dB) on every level, with ample speed room (at most 0.316 of SLEEF, 0.218 of liquid-dsp); `deemphasis_fallback` 145.36 against 151.98 dB (6.6 dB), speed room small on the reference models (0.920 `skylake`, 0.945 `znver4`, 0.884 `znver3`, 0.833 `cortex-a72`); `fir_decimate_fallback` 133.29 against 134.33 dB (1.04 dB), speed room except `neon` (0.961 on `cortex-a72`); `am_mod_fallback` at `neon` 150.80 against 152.07 dB (1.27 dB, and 2.27 times slower, group 1).

### Group 3a: the Q15 / CQ15 FIR SIMD entries (D65 switch), ours / own fallback

| Entry | Level | Ours (cycles per sample, ref. / counter) | Library (cycles per sample, ref. / counter) | Ours / library (ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback (ref. / counter) | Models (ref. / counter) | SNR in dB (ours / library) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fir_cq15` | `sse2` | 142.26 / 188.05 | — / — | — / — | 142.26 / 188.05 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `fir_cq15` | `sse4.2` | 47.76 / 49.10 | — / — | — / — | 142.26 / 188.05 | 0.336 / 0.261 | skylake / znver2 | 999.00 / — | `info` |
| `fir_cq15` | `avx2` | 40.45 / 60.95 | — / — | — / — | 40.45 / 60.95 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / — | `info` |
| `fir_cq15` | `avx2-runtime` | 41.11 / 61.89 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `fir_cq15` | `avx512` | 40.37 / 45.20 | — / — | — / — | 40.37 / 45.20 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `fir_cq15` | `neon` | 106.10 / 92.51 | — / — | — / — | 107.54 / 97.99 | 0.987 / 0.944 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |
| `fir_decimate_cq15` | `sse2` | 28.57 / 34.29 | — / — | — / — | 28.57 / 34.29 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `fir_decimate_cq15` | `sse4.2` | 8.87 / 9.34 | — / — | — / — | 35.07 / 28.09 | 0.253 / 0.333 | skylake / znver2 | 999.00 / — | `info` |
| `fir_decimate_cq15` | `avx2` | 21.61 / 22.05 | — / — | — / — | 21.61 / 22.05 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / — | `info` |
| `fir_decimate_cq15` | `avx2-runtime` | 18.97 / 20.01 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `fir_decimate_cq15` | `avx512` | 11.57 / 11.17 | — / — | — / — | 16.80 / 18.22 | 0.689 / 0.613 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `fir_decimate_cq15` | `neon` | 14.98 / 14.64 | — / — | — / — | 14.98 / 14.64 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |
| `fir_interpolate_cq15` | `sse2` | 137.49 / 207.95 | — / — | — / — | 137.49 / 207.95 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `fir_interpolate_cq15` | `sse4.2` | 51.44 / 53.59 | — / — | — / — | 137.49 / 207.95 | 0.374 / 0.258 | skylake / znver2 | 999.00 / — | `info` |
| `fir_interpolate_cq15` | `avx2` | 52.53 / 71.44 | — / — | — / — | 52.53 / 71.44 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / — | `info` |
| `fir_interpolate_cq15` | `avx2-runtime` | 52.41 / 71.69 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `fir_interpolate_cq15` | `avx512` | 54.43 / 63.07 | — / — | — / — | 54.43 / 63.07 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `fir_interpolate_cq15` | `neon` | 75.80 / 76.00 | — / — | — / — | 127.87 / 145.98 | 0.593 / 0.521 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |
| `resample_cq15` | `sse2` | 168.72 / 240.51 | — / — | — / — | 168.72 / 240.51 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `resample_cq15` | `sse4.2` | 49.95 / 51.77 | — / — | — / — | 168.72 / 240.51 | 0.296 / 0.215 | skylake / znver2 | 999.00 / — | `info` |
| `resample_cq15` | `avx2` | 117.51 / 113.18 | — / — | — / — | 117.51 / 113.18 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / — | `info` |
| `resample_cq15` | `avx2-runtime` | 97.21 / 110.41 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `resample_cq15` | `avx512` | 66.27 / 68.77 | — / — | — / — | 84.14 / 88.98 | 0.788 / 0.773 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `resample_cq15` | `neon` | 145.75 / 118.45 | — / — | — / — | 193.78 / 184.91 | 0.752 / 0.641 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |
| `fir_interpolate_real_q15` | `sse2` | 70.71 / 101.24 | — / — | — / — | 70.71 / 101.24 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `fir_interpolate_real_q15` | `sse4.2` | 26.58 / 32.31 | — / — | — / — | 70.71 / 101.24 | 0.376 / 0.319 | skylake / znver2 | 999.00 / — | `info` |
| `fir_interpolate_real_q15` | `avx2` | 42.13 / 38.06 | — / — | — / — | 42.13 / 38.06 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / — | `info` |
| `fir_interpolate_real_q15` | `avx2-runtime` | 43.09 / 40.75 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `fir_interpolate_real_q15` | `avx512` | 22.72 / 29.27 | — / — | — / — | 22.72 / 29.27 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `fir_interpolate_real_q15` | `neon` | 73.09 / 73.97 | — / — | — / — | 73.09 / 73.97 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |
| `resample_real_q15` | `sse2` | 103.67 / 144.16 | — / — | — / — | 103.67 / 144.16 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `resample_real_q15` | `sse4.2` | 25.84 / 31.40 | — / — | — / — | 103.67 / 144.16 | 0.249 / 0.218 | skylake / znver2 | 999.00 / — | `info` |
| `resample_real_q15` | `avx2` | 60.80 / 66.72 | — / — | — / — | 112.73 / 99.45 | 0.539 / 0.671 | znver3 / alderlake | 999.00 / — | `info` |
| `resample_real_q15` | `avx2-runtime` | 60.54 / 66.87 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `resample_real_q15` | `avx512` | 33.26 / 36.27 | — / — | — / — | 100.33 / 85.43 | 0.332 / 0.425 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `resample_real_q15` | `neon` | 97.68 / 78.50 | — / — | — / — | 113.65 / 107.66 | 0.860 / 0.729 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |

`simd128` (instructions per sample):

| Entry | Ours (instructions per sample) | Judged library (instructions per sample) | VOLK `-O3 -msimd128` (instructions per sample) | Ours / judged | Ours / stronger (D2953) | Own fallback (instructions per sample) | Ours / own fallback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fir_cq15` | 841.33 | — | — | — | — | 934.94 | 0.900 | `info` |
| `fir_decimate_cq15` | 185.86 | — | — | — | — | 205.53 | 0.904 | `info` |
| `fir_interpolate_cq15` | 638.42 | — | — | — | — | 638.42 | 1.000 | `info` |
| `resample_cq15` | 976.93 | — | — | — | — | 1075.66 | 0.908 | `info` |
| `fir_interpolate_real_q15` | 451.84 | — | — | — | — | 451.84 | 1.000 | `info` |
| `resample_real_q15` | 598.51 | — | — | — | — | 875.80 | 0.683 | `info` |

Target: at most 0.80 of the fallback on both models at every level. Above it today (1.000 is the D65 switch): `sse2` all six; `avx2` all but `resample_real_q15` (0.539 / 0.671); `avx512` `fir_cq15`, `fir_interpolate_cq15`, `fir_interpolate_real_q15` (switched), `resample_cq15` 0.788 / 0.773 (just within); `neon` `fir_decimate_cq15` and `fir_interpolate_real_q15` (switched), `fir_cq15` 0.987 / 0.944 (kept, no gain), `resample_real_q15` 0.860 / 0.729; `simd128` `fir_interpolate_cq15` and `fir_interpolate_real_q15` (switched, 1.00), `fir_cq15` and `fir_decimate_cq15` 0.90, `resample_cq15` 0.91. Within the target: every `sse4.2` row (0.215 to 0.376, the `pmaddwd` progressions of D340), `fir_decimate_cq15` at `avx512`, `fir_interpolate_cq15` and `resample_cq15` at `neon`, `resample_real_q15` at `avx2`, `avx512` and `simd128`. `avx2-runtime`: `fir_interpolate_real_q15` 43.09 / 40.75 against 42.13 / 38.06 of the fixed `avx2` build (1.023 / 1.071, information, group 4).

### Group 3a: Q15 kernels without a SIMD branch at sse2 (the sse4.2 row shows what the SSE4.1/SSSE3 branch gives)

| Entry | Level | Ours (cycles per sample, ref. / counter) | Library (cycles per sample, ref. / counter) | Ours / library (ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback (ref. / counter) | Models (ref. / counter) | SNR in dB (ours / library) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `am_mod_cq15` | `sse2` | 2.62 / 3.12 | — / — | — / — | 2.62 / 3.12 | 1.000 / 1.000 | skylake / znver2 | 92.65 / — | `info` |
| `am_mod_cq15` | `sse4.2` | 0.31 / 0.37 | — / — | — / — | 1.23 / 1.15 | 0.250 / 0.318 | skylake / znver2 | 92.65 / — | `info` |
| `biquad_cq15` | `sse2` | 24.94 / 31.17 | — / — | — / — | 24.94 / 31.17 | 1.000 / 1.000 | skylake / znver2 | 71.59 / — | `info` |
| `biquad_cq15` | `sse4.2` | 16.13 / 15.63 | — / — | — / — | 24.43 / 31.16 | 0.660 / 0.501 | skylake / znver2 | 71.59 / — | `info` |
| `biquad_real_q15` | `sse2` | 14.12 / 15.13 | — / — | — / — | 14.12 / 15.13 | 1.000 / 1.000 | skylake / znver2 | 71.45 / — | `info` |
| `biquad_real_q15` | `sse4.2` | 12.08 / 11.32 | — / — | — / — | 14.13 / 15.13 | 0.855 / 0.749 | skylake / znver2 | 71.45 / — | `info` |
| `mul_cq15` | `sse2` | 5.03 / 3.90 | — / — | — / — | 5.03 / 3.90 | 1.000 / 1.000 | skylake / znver2 | 94.36 / — | `info` |
| `mul_cq15` | `sse4.2` | 1.43 / 1.48 | — / — | — / — | 3.04 / 2.89 | 0.469 / 0.514 | skylake / znver2 | 94.36 / — | `info` |
| `mul_real_q15` | `sse2` | 0.97 / 1.02 | — / — | — / — | 0.97 / 1.02 | 1.000 / 1.000 | skylake / znver2 | 91.62 / — | `info` |
| `mul_real_q15` | `sse4.2` | 0.20 / 0.24 | — / — | — / — | 0.67 / 0.55 | 0.295 / 0.427 | skylake / znver2 | 91.62 / — | `info` |
| `scale_cq15` | `sse2` | 2.03 / 2.11 | — / — | — / — | 2.03 / 2.11 | 1.000 / 1.000 | skylake / znver2 | 93.67 / — | `info` |
| `scale_cq15` | `sse4.2` | 0.26 / 0.26 | — / — | — / — | 1.10 / 1.02 | 0.237 / 0.256 | skylake / znver2 | 93.67 / — | `info` |

`simd128` (instructions per sample):

| Entry | Ours (instructions per sample) | Judged library (instructions per sample) | VOLK `-O3 -msimd128` (instructions per sample) | Ours / judged | Ours / stronger (D2953) | Own fallback (instructions per sample) | Ours / own fallback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mul_cq15` | 15.17 | — | — | — | — | 15.17 | 1.000 | `info` |

All six run the fallback at `sse2` (1.000); the `sse4.2` branches give 0.237 to 0.660 of the fallback, `biquad_real_q15` 0.855 / 0.749 (above 0.80 on `skylake` at `sse4.2` as well). `mul_cq15` has no `simd128` path either (1.00). The six FIR kernels at `sse2` are in the table above.

### Group 3b: SIMD entries no faster than their fallback

| Entry | Level | Ours (cycles per sample, ref. / counter) | Library (cycles per sample, ref. / counter) | Ours / library (ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback (ref. / counter) | Models (ref. / counter) | SNR in dB (ours / library) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `am_mod` | `sse2` | 1.09 / 1.16 | — / — | — / — | 1.09 / 1.16 | 1.000 / 1.000 | skylake / znver2 | 150.80 / — | `info` |
| `am_mod` | `sse4.2` | 1.09 / 1.16 | — / — | — / — | 1.09 / 1.16 | 1.000 / 1.000 | skylake / znver2 | 150.80 / — | `info` |
| `am_mod` | `avx2` | 0.65 / 0.47 | — / — | — / — | 0.65 / 0.47 | 1.000 / 1.000 | znver3 / alderlake | 150.80 / — | `info` |
| `am_mod` | `avx2-runtime` | 0.65 / 0.47 | — / — | — / — | — / — | — / — | znver3 / alderlake | 150.80 / — | `info` |
| `am_mod` | `avx512` | 0.38 / 0.30 | — / — | — / — | 0.38 / 0.30 | 1.000 / 1.000 | znver4 / sapphirerapids | 150.80 / — | `info` |
| `am_mod` | `neon` | 2.91 / 1.52 | — / — | — / — | 2.91 / 1.52 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 150.80 / — | `info` |
| `am_mod_cq15` | `sse2` | 2.62 / 3.12 | — / — | — / — | 2.62 / 3.12 | 1.000 / 1.000 | skylake / znver2 | 92.65 / — | `info` |
| `am_mod_cq15` | `sse4.2` | 0.31 / 0.37 | — / — | — / — | 1.23 / 1.15 | 0.250 / 0.318 | skylake / znver2 | 92.65 / — | `info` |
| `am_mod_cq15` | `avx2` | 0.20 / 0.21 | — / — | — / — | 0.43 / 0.56 | 0.473 / 0.368 | znver3 / alderlake | 92.65 / — | `info` |
| `am_mod_cq15` | `avx2-runtime` | 0.20 / 0.21 | — / — | — / — | — / — | — / — | znver3 / alderlake | 92.65 / — | `info` |
| `am_mod_cq15` | `avx512` | 0.14 / 0.20 | — / — | — / — | 0.32 / 0.48 | 0.431 / 0.426 | znver4 / sapphirerapids | 92.65 / — | `info` |
| `am_mod_cq15` | `neon` | 0.90 / 0.77 | — / — | — / — | 1.72 / 1.33 | 0.524 / 0.576 | cortex-a72 / cortex-a76 | 92.65 / — | `info` |
| `mag` | `avx512` | 0.64 / 0.76 | 1.28 / 1.54 | 0.496 / 0.494 | 0.64 / 0.76 | 0.999 / 0.999 | znver4 / sapphirerapids | 150.54 / 150.92 | `release` |
| `mag` | `neon` | 8.58 / 1.77 | 8.59 / 1.79 | 0.998 / 0.988 | 8.56 / 1.77 | 1.002 / 0.997 | cortex-a72 / cortex-a76 | 150.54 / 150.95 | `release` |
| `mul` | `sse2` | 1.53 / 1.59 | 2.68 / 2.81 | 0.570 / 0.567 | 1.53 / 1.59 | 1.001 / 1.001 | skylake / znver2 | 148.50 / 148.50 | `release` |
| `stereo_matrix` | `neon` | 1.90 / 1.39 | 5.74 / 5.23 | 0.331 / 0.267 | 1.90 / 1.39 | 1.001 / 1.001 | cortex-a72 / cortex-a76 | 151.28 / 151.28 | `release` |
| `stereo_matrix_cq15` | `neon` | 0.77 / 0.76 | — / — | — / — | 0.77 / 0.76 | 1.002 / 1.002 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |
| `dot_cq15` | `neon` | 0.90 / 0.90 | — / — | — / — | 0.90 / 0.90 | 0.999 / 1.000 | cortex-a72 / cortex-a76 | 82.20 / — | `info` |
| `interleave` | `avx2` | 0.26 / 0.51 | 0.27 / 0.52 | 0.971 / 0.974 | 0.26 / 0.51 | 0.995 / 0.993 | znver3 / alderlake | 999.00 / 999.00 | `release` |
| `interleave_cq15` | `avx2` | 0.14 / 0.13 | 6.55 / 7.64 | 0.021 / 0.017 | 0.13 / 0.13 | 1.010 / 0.999 | znver3 / alderlake | 999.00 / 999.00 | `release` |
| `deinterleave` | `avx2` | 0.26 / 0.26 | 0.39 / 0.40 | 0.662 / 0.651 | 0.26 / 0.26 | 0.991 / 0.984 | znver3 / alderlake | 999.00 / 999.00 | `release` |
| `convert_cq15_cs8` | `avx2` | 0.14 / 0.14 | 0.14 / 0.18 | 0.967 / 0.764 | 0.14 / 0.14 | 0.992 / 0.986 | znver3 / alderlake | 999.00 / 999.00 | `release` |
| `equalize` | `sse2` | 138.46 / 145.11 | 355.18 / 390.47 | 0.390 / 0.372 | 138.46 / 145.11 | 1.000 / 1.000 | skylake / znver2 | 999.00 / 129.96 | `info` |
| `equalize` | `sse4.2` | 135.40 / 133.75 | 271.02 / 292.38 | 0.500 / 0.457 | 135.40 / 133.75 | 1.000 / 1.000 | skylake / znver2 | 999.00 / 129.82 | `info` |
| `equalize` | `avx2` | 111.94 / 152.23 | 1074.99 / 289.89 | 0.104 / 0.525 | 111.94 / 152.23 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / 130.26 | `info` |
| `equalize` | `avx2-runtime` | 105.10 / 100.09 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `equalize` | `avx512` | 104.01 / 146.19 | 1094.85 / 323.95 | 0.095 / 0.451 | 104.01 / 146.19 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / 130.26 | `info` |
| `equalize` | `neon` | 216.21 / 183.98 | 329.80 / 359.68 | 0.656 / 0.512 | 216.21 / 183.98 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 999.00 / 129.52 | `info` |
| `equalize_qam16` | `sse2` | 159.59 / 158.25 | 357.76 / 396.73 | 0.446 / 0.399 | 159.59 / 158.25 | 1.000 / 1.000 | skylake / znver2 | 999.00 / 129.81 | `info` |
| `equalize_qam16` | `sse4.2` | 160.09 / 152.75 | 273.34 / 297.82 | 0.586 / 0.513 | 160.09 / 152.75 | 1.000 / 1.000 | skylake / znver2 | 999.00 / 129.85 | `info` |
| `equalize_qam16` | `avx2` | 119.94 / 164.27 | 1088.28 / 294.40 | 0.110 / 0.558 | 119.94 / 164.27 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / 130.02 | `info` |
| `equalize_qam16` | `avx2-runtime` | 129.16 / 111.96 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `equalize_qam16` | `avx512` | 121.27 / 169.39 | 1100.18 / 328.53 | 0.110 / 0.516 | 121.27 / 169.39 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / 130.02 | `info` |
| `equalize_qam16` | `neon` | 229.29 / 190.02 | 357.37 / 388.90 | 0.642 / 0.489 | 229.29 / 190.02 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 999.00 / 130.13 | `info` |
| `equalize_qam64` | `sse2` | 159.59 / 158.25 | 357.76 / 396.73 | 0.446 / 0.399 | 159.59 / 158.25 | 1.000 / 1.000 | skylake / znver2 | 999.00 / 130.61 | `info` |
| `equalize_qam64` | `sse4.2` | 160.09 / 152.75 | 273.34 / 297.82 | 0.586 / 0.513 | 160.09 / 152.75 | 1.000 / 1.000 | skylake / znver2 | 999.00 / 130.57 | `info` |
| `equalize_qam64` | `avx2` | 123.94 / 170.30 | 1088.28 / 294.40 | 0.114 / 0.578 | 123.94 / 170.30 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / 130.82 | `info` |
| `equalize_qam64` | `avx2-runtime` | 129.16 / 112.47 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `equalize_qam64` | `avx512` | 121.24 / 168.37 | 1100.18 / 328.53 | 0.110 / 0.512 | 121.24 / 168.37 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / 130.82 | `info` |
| `equalize_qam64` | `neon` | 244.37 / 189.01 | 357.37 / 388.90 | 0.684 / 0.486 | 244.37 / 189.01 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 999.00 / 131.03 | `info` |
| `equalize_qam256` | `sse2` | 160.64 / 164.28 | — / — | — / — | 160.64 / 164.28 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `equalize_qam256` | `sse4.2` | 159.18 / 158.86 | — / — | — / — | 159.18 / 158.86 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `equalize_qam256` | `avx2` | 142.06 / 174.29 | — / — | — / — | 142.06 / 174.29 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / — | `info` |
| `equalize_qam256` | `avx2-runtime` | 134.21 / 116.33 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `equalize_qam256` | `avx512` | 125.31 / 173.38 | — / — | — / — | 125.31 / 173.38 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `equalize_qam256` | `neon` | 252.46 / 186.09 | — / — | — / — | 252.46 / 186.09 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |
| `equalize_qam1024` | `sse2` | 160.64 / 164.28 | — / — | — / — | 160.64 / 164.28 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `equalize_qam1024` | `sse4.2` | 159.18 / 158.86 | — / — | — / — | 159.18 / 158.86 | 1.000 / 1.000 | skylake / znver2 | 999.00 / — | `info` |
| `equalize_qam1024` | `avx2` | 142.06 / 174.29 | — / — | — / — | 142.06 / 174.29 | 1.000 / 1.000 | znver3 / alderlake | 999.00 / — | `info` |
| `equalize_qam1024` | `avx2-runtime` | 134.21 / 116.33 | — / — | — / — | — / — | — / — | znver3 / alderlake | 999.00 / — | `info` |
| `equalize_qam1024` | `avx512` | 125.31 / 173.38 | — / — | — / — | 125.31 / 173.38 | 1.000 / 1.000 | znver4 / sapphirerapids | 999.00 / — | `info` |
| `equalize_qam1024` | `neon` | 252.46 / 186.09 | — / — | — / — | 252.46 / 186.09 | 1.000 / 1.000 | cortex-a72 / cortex-a76 | 999.00 / — | `info` |

`simd128` (instructions per sample):

| Entry | Ours (instructions per sample) | Judged library (instructions per sample) | VOLK `-O3 -msimd128` (instructions per sample) | Ours / judged | Ours / stronger (D2953) | Own fallback (instructions per sample) | Ours / own fallback | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `am_mod` | 10.54 | — | — | — | — | 10.54 | 1.000 | `info` |
| `am_mod_cq15` | 3.92 | — | — | — | — | 8.80 | 0.445 | `info` |
| `equalize` | 1017.52 | — | — | — | — | 1017.52 | 1.000 | `info` |
| `equalize_qam16` | 1065.53 | — | — | — | — | 1065.53 | 1.000 | `info` |
| `equalize_qam64` | 1064.53 | — | — | — | — | 1064.53 | 1.000 | `info` |
| `equalize_qam256` | 1020.53 | — | — | — | — | 1020.53 | 1.000 | `info` |
| `equalize_qam1024` | 1020.53 | — | — | — | — | 1020.53 | 1.000 | `info` |

At 1.000 (the same code or within 1 %): `am_mod` on every level (no branch); `mag` at `avx512` (0.999 / 0.999) and `neon` (1.002 / 0.997); `mul` at `sse2` (1.001 / 1.001); `stereo_matrix` and `stereo_matrix_cq15` at `neon` (1.001, 1.002); `dot_cq15` at `neon` (0.999 / 1.000); `interleave` (0.995 / 0.993), `interleave_cq15` (1.010 / 0.999), `deinterleave` (0.991 / 0.984) and `convert_cq15_cs8` (0.992 / 0.986) at `avx2`; `equalize` and the four QAM `equalize` entries on every level (the fallback's loop, D841). `am_mod_cq15` is within the target except at `sse2` (group 3a).

### Group 4: above 1.10 of the library

| Entry | Level | Ours (cycles per sample, ref. / counter) | Library (cycles per sample, ref. / counter) | Ours / library (ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback (ref. / counter) | Models (ref. / counter) | SNR in dB (ours / library) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fft_cq15_1024` | `neon` | 11.77 / 10.10 | 8.91 / 10.01 | 1.320 / 1.009 | 120.91 / 100.37 | 0.097 / 0.101 | cortex-a72 / cortex-a76 | 63.54 / 44.79 | `dev` |
| `fft_cq15_1024_full_scale` | `neon` | 10.93 / 9.01 | 8.91 / 10.01 | 1.227 / 0.900 | — / — | — / — | cortex-a72 / cortex-a76 | 59.88 / 56.90 | `dev` |
| `convert_out_cu8_fallback` | `sse2` | 5.43 / 5.74 | 12.29 / 14.25 | 0.442 / 0.402 | — / — | — / — | skylake / znver2 | 47.27 / 47.27 | `release` |
| `convert_out_cu8_fallback` | `sse4.2` | 4.70 / 4.41 | 2.83 / 2.72 | 1.658 / 1.621 | — / — | — / — | skylake / znver2 | 47.27 / 47.27 | `dev` |
| `convert_out_cu8_fallback` | `avx2` | 1.67 / 1.82 | 1.22 / 1.49 | 1.371 / 1.224 | — / — | — / — | znver3 / alderlake | 47.27 / 47.27 | `dev` |
| `convert_out_cu8_fallback` | `avx512` | 1.22 / 1.33 | 0.68 / 1.27 | 1.805 / 1.041 | — / — | — / — | znver4 / sapphirerapids | 47.27 / 47.27 | `dev` |
| `convert_out_cu8_fallback` | `neon` | 7.14 / 4.35 | 4.90 / 2.84 | 1.456 / 1.532 | — / — | — / — | cortex-a72 / cortex-a76 | 47.27 / 47.27 | `dev` |
| `convert_out_real_ru8_fallback` | `sse2` | 2.72 / 2.88 | 6.15 / 7.13 | 0.443 / 0.404 | — / — | — / — | skylake / znver2 | 47.25 / 47.25 | `release` |
| `convert_out_real_ru8_fallback` | `sse4.2` | 2.36 / 2.21 | 1.42 / 1.36 | 1.661 / 1.623 | — / — | — / — | skylake / znver2 | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8_fallback` | `avx2` | 0.84 / 0.92 | 0.61 / 0.75 | 1.372 / 1.226 | — / — | — / — | znver3 / alderlake | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8_fallback` | `avx512` | 0.61 / 0.67 | 0.34 / 0.64 | 1.797 / 1.046 | — / — | — / — | znver4 / sapphirerapids | 47.25 / 47.25 | `dev` |
| `convert_out_real_ru8_fallback` | `neon` | 3.58 / 2.18 | 2.46 / 1.42 | 1.457 / 1.533 | — / — | — / — | cortex-a72 / cortex-a76 | 47.25 / 47.25 | `dev` |
| `convert_out_cs8_fallback` | `sse2` | 2.32 / 2.48 | 11.72 / 14.21 | 0.198 / 0.175 | — / — | — / — | skylake / znver2 | 47.31 / 47.31 | `release` |
| `convert_out_cs8_fallback` | `sse4.2` | 1.91 / 1.88 | 2.77 / 2.52 | 0.688 / 0.747 | — / — | — / — | skylake / znver2 | 47.31 / 47.31 | `release` |
| `convert_out_cs8_fallback` | `avx2` | 1.06 / 1.02 | 1.17 / 1.43 | 0.911 / 0.717 | — / — | — / — | znver3 / alderlake | 47.31 / 47.31 | `release` |
| `convert_out_cs8_fallback` | `avx512` | 0.73 / 0.84 | 0.56 / 1.27 | 1.310 / 0.663 | — / — | — / — | znver4 / sapphirerapids | 47.31 / 47.31 | `dev` |
| `convert_out_cs8_fallback` | `neon` | 5.80 / 3.09 | 4.47 / 2.65 | 1.299 / 1.168 | — / — | — / — | cortex-a72 / cortex-a76 | 47.31 / 47.31 | `dev` |
| `convert_out_real_rs8_fallback` | `sse2` | 1.17 / 1.25 | 5.86 / 7.11 | 0.199 / 0.175 | — / — | — / — | skylake / znver2 | 47.24 / 47.24 | `release` |
| `convert_out_real_rs8_fallback` | `sse4.2` | 0.96 / 0.94 | 1.39 / 1.26 | 0.691 / 0.750 | — / — | — / — | skylake / znver2 | 47.24 / 47.24 | `release` |
| `convert_out_real_rs8_fallback` | `avx2` | 0.54 / 0.52 | 0.59 / 0.72 | 0.916 / 0.723 | — / — | — / — | znver3 / alderlake | 47.24 / 47.24 | `release` |
| `convert_out_real_rs8_fallback` | `avx512` | 0.37 / 0.43 | 0.28 / 0.64 | 1.316 / 0.670 | — / — | — / — | znver4 / sapphirerapids | 47.24 / 47.24 | `dev` |
| `convert_out_real_rs8_fallback` | `neon` | 2.91 / 1.55 | 2.24 / 1.33 | 1.300 / 1.170 | — / — | — / — | cortex-a72 / cortex-a76 | 47.24 / 47.24 | `dev` |
| `stereo_matrix_fallback` | `sse2` | 1.55 / 1.67 | 1.07 / 1.38 | 1.445 / 1.211 | — / — | — / — | skylake / znver2 | 151.28 / 151.28 | `dev` |
| `stereo_matrix_fallback` | `sse4.2` | 1.19 / 1.34 | 1.07 / 1.38 | 1.116 / 0.969 | — / — | — / — | skylake / znver2 | 151.28 / 151.28 | `dev` |
| `stereo_matrix_fallback` | `avx2` | 0.44 / 0.65 | 0.60 / 0.53 | 0.737 / 1.211 | — / — | — / — | znver3 / alderlake | 151.28 / 151.28 | `dev` |
| `stereo_matrix_fallback` | `avx512` | 0.48 / 0.54 | 0.60 / 0.53 | 0.800 / 1.020 | — / — | — / — | znver4 / sapphirerapids | 151.28 / 151.28 | `release` |
| `stereo_matrix_fallback` | `neon` | 1.90 / 1.39 | 3.83 / 3.07 | 0.495 / 0.453 | — / — | — / — | cortex-a72 / cortex-a76 | 151.28 / 151.28 | `release` |
| `fir_decimate_cq15_fallback` | `sse2` | 28.57 / 34.29 | 24.93 / 25.79 | 1.146 / 1.330 | — / — | — / — | skylake / znver2 | 999.00 / 82.19 | `dev` |
| `fir_decimate_cq15_fallback` | `sse4.2` | 35.07 / 28.09 | 27.95 / 22.79 | 1.255 / 1.232 | — / — | — / — | skylake / znver2 | 999.00 / 82.19 | `dev` |
| `fir_decimate_cq15_fallback` | `avx2` | 21.61 / 22.05 | 42.02 / 17.52 | 0.514 / 1.258 | — / — | — / — | znver3 / alderlake | 999.00 / 82.19 | `dev` |
| `fir_decimate_cq15_fallback` | `avx512` | 16.80 / 18.22 | 37.21 / 20.44 | 0.451 / 0.891 | — / — | — / — | znver4 / sapphirerapids | 999.00 / 82.19 | `release` |
| `fir_interpolate_real_q15_fallback` | `sse2` | 70.71 / 101.24 | 68.81 / 82.21 | 1.028 / 1.231 | — / — | — / — | skylake / znver2 | 999.00 / 80.35 | `dev` |
| `fir_interpolate_real_q15_fallback` | `sse4.2` | 70.71 / 101.24 | 68.81 / 82.21 | 1.028 / 1.231 | — / — | — / — | skylake / znver2 | 999.00 / 80.35 | `dev` |
| `fir_interpolate_cq15_fallback` | `sse2` | 137.49 / 207.95 | 137.61 / 164.43 | 0.999 / 1.265 | — / — | — / — | skylake / znver2 | 999.00 / 80.09 | `dev` |
| `fir_interpolate_cq15_fallback` | `sse4.2` | 137.49 / 207.95 | 137.61 / 164.43 | 0.999 / 1.265 | — / — | — / — | skylake / znver2 | 999.00 / 80.09 | `dev` |
| `dot_cq15_fallback` | `sse2` | 2.20 / 2.53 | 1.78 / 2.03 | 1.232 / 1.244 | — / — | — / — | skylake / znver2 | 82.20 / 82.20 | `dev` |
| `dot_cq15_fallback` | `sse4.2` | 2.03 / 1.72 | 2.04 / 1.49 | 0.995 / 1.153 | — / — | — / — | skylake / znver2 | 82.20 / 82.20 | `dev` |
| `mul_real_q15_fallback` | `avx2` | 0.28 / 0.39 | 0.30 / 0.31 | 0.924 / 1.250 | — / — | — / — | znver3 / alderlake | 91.62 / 85.59 | `dev` |
| `deinterleave_cq15_fallback` | `avx2` | 0.23 / 0.23 | 0.20 / 0.26 | 1.135 / 0.860 | — / — | — / — | znver3 / alderlake | 999.00 / 999.00 | `dev` |
| `mag_sq_fallback` | `avx512` | 0.37 / 0.26 | 0.31 / 0.27 | 1.210 / 0.942 | — / — | — / — | znver4 / sapphirerapids | 149.02 / 149.02 | `dev` |
| `mul_fallback` | `avx512` | 0.77 / 0.78 | 0.53 / 0.79 | 1.466 / 0.977 | — / — | — / — | znver4 / sapphirerapids | 148.50 / 148.50 | `dev` |
| `scramble_c32_b8` | `avx512` | 0.58 / 0.90 | 2.55 / 2.39 | 0.228 / 0.378 | 0.95 / 3.92 | 0.611 / 0.230 | znver4 / sapphirerapids | 147.72 / 147.72 | `release` |
| `mul` | `avx512` | 0.58 / 0.39 | 0.71 / 0.53 | 0.816 / 0.733 | 0.77 / 0.78 | 0.746 / 0.497 | znver4 / sapphirerapids | 150.12 / 148.50 | `release` |
| `channelize_fallback` | `sse2` | 59.64 / 62.71 | 166.47 / 182.49 | 0.358 / 0.344 | — / — | — / — | skylake / znver2 | 140.65 / 140.76 | `release` |
| `channelize_fallback` | `sse4.2` | 60.03 / 62.59 | 166.73 / 182.23 | 0.360 / 0.343 | — / — | — / — | skylake / znver2 | 140.65 / 140.76 | `release` |
| `channelize_fallback` | `avx2` | 37.13 / 34.35 | 147.96 / 133.26 | 0.251 / 0.258 | — / — | — / — | znver3 / alderlake | 140.65 / 140.76 | `release` |
| `channelize_fallback` | `avx512` | 40.17 / 34.30 | 105.93 / 118.74 | 0.379 / 0.289 | — / — | — / — | znver4 / sapphirerapids | 140.65 / 140.76 | `release` |
| `channelize_fallback` | `neon` | 97.98 / 80.80 | 221.93 / 165.45 | 0.441 / 0.488 | — / — | — / — | cortex-a72 / cortex-a76 | 140.65 / 140.76 | `release` |

Above 1.10 on a model: `fft_cq15_1024` (1.320 on `cortex-a72`) and `fft_cq15_1024_full_scale` (1.227) at `neon` against Ne10; the 8-bit `convert_out` fallbacks against VOLK `generic`: `cu8` and `real_ru8` at `sse4.2` (1.66 / 1.62, above 1.50), `avx2` (1.37 / 1.22), `avx512` (1.81 / 1.04, above 1.50 on `znver4`), `neon` (1.46 / 1.53, above 1.50 on `cortex-a76`), `cs8` and `real_rs8` at `avx512` (1.31 on `znver4`) and `neon` (1.30 / 1.17); `stereo_matrix_fallback` at `sse2` (1.445 / 1.211), `sse4.2` (1.116) and `avx2` (1.211 on `alderlake`); `fir_decimate_cq15_fallback` at `sse2` (1.146 / 1.330), `sse4.2` (1.255 / 1.232), `avx2` (1.258 on `alderlake`); `fir_interpolate_real_q15_fallback` and `fir_interpolate_cq15_fallback` at `sse2` and `sse4.2` on `znver2` (1.231, 1.265); `dot_cq15_fallback` at `sse2` (1.232 / 1.244) and `sse4.2` (1.153 on `znver2`); `mul_real_q15_fallback` at `avx2` (1.250 on `alderlake`); `deinterleave_cq15_fallback` at `avx2` (1.135 on `znver3`); `mag_sq_fallback` at `avx512` (1.210 on `znver4`); `mul_fallback` at `avx512` (1.466 on `znver4`, D2951). Information: IO1-1 stays closed (`mul` at `avx512` `release`, 0.816 / 0.733); IO1-8 (`scramble_c32_b8` at `avx512`, 0.228 / 0.378) is a question of the `sapphirerapids` model, not of a ratio; `channelize_fallback` is `release` at 0.25 to 0.49 of liquid-dsp, its 4.5 to 7.6 % of D2811 / D2841 are against its own earlier code.

### Outside the agreed groups (for the maintainer)

The sweep of the same records finds `dev` entries above 1.10 of their library that no group names: at `simd128` (instructions per sample, against KFR's wasm32 build) `fft_4096_fallback` 1.107, `fir_cq15_fallback` 1.190, `fir_interpolate_real_fallback` 1.184, `fir_interpolate_short_fallback` 1.385, `scale_fallback` 1.141, `stereo_pll_fallback` 1.314; at `neon` `fir_interpolate_real_fallback` 1.224 and `fir_interpolate_short_fallback` 1.133 on `cortex-a72` (rows of I13.1-7 that O1 left). D2953 changes no `simd128` status: VOLK's generic C with `-O3 -msimd128` is the stronger opponent of 17 entries (`add`, `add_real`, `mul_real`, `mag`, `mag_sq`, `dot`, `window`, `stereo_matrix`, the (de)interleaving and the conversions to `C32` / `CQ15`), and against it all 17 stay `release` (at most 0.816, `add`).

### Group 5 preview: SIMD entries above 0.80 of their own fallback

The report that group 5 adds to `cargo xtask check cycles`, computed from the same records for the kernel entries (blocks, protocol coders and the `halfband_ref_*` entries left out). Most of the element-wise rows (0.80 to 1.00) are bound by loads and stores, where the compiler vectorises the fallback as well; the control loops (`costas*`, `symbol_sync`, `agc` at `neon`) are O3.

62 kernel entries, 221 entry-levels

| SIMD entry | Levels above 0.80 of the own fallback (ours / own fallback, ref. / counter model; `simd128` instructions) |
| --- | --- |
| `add` | `avx2` 0.99 / 0.95; `avx512` 0.96 / 0.74; `neon` 0.93 / 0.91; `simd128` 0.92 |
| `add_cq15` | `avx2` 0.99 / 0.96; `avx512` 0.97 / 0.75; `neon` 0.92 / 0.92; `simd128` 1.00 |
| `add_real` | `sse2` 0.87 / 0.78; `sse4.2` 0.87 / 0.78; `avx2` 0.99 / 0.95 |
| `add_real_q15` | `sse2` 0.88 / 0.79; `sse4.2` 0.88 / 0.79; `avx2` 0.99 / 0.96; `simd128` 1.00 |
| `agc` | `avx2` 0.74 / 0.80; `avx512` 0.71 / 0.83; `neon` 1.00 / 1.00; `simd128` 1.01 |
| `am_mod` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `am_mod_cq15` | `sse2` 1.00 / 1.00 |
| `biquad_cq15` | `sse2` 1.00 / 1.00 |
| `biquad_real` | `sse2` 0.86 / 0.80; `sse4.2` 0.83 / 0.80; `avx2` 0.90 / 0.86; `avx512` 0.96 / 0.90; `neon` 0.93 / 0.73 |
| `biquad_real_q15` | `sse2` 1.00 / 1.00; `sse4.2` 0.85 / 0.75; `avx2` 0.74 / 0.84; `avx512` 0.75 / 0.81; `neon` 0.94 / 0.46 |
| `convert_cq15_cs8` | `sse2` 0.87 / 0.73; `sse4.2` 0.86 / 0.88; `avx2` 0.99 / 0.99; `neon` 0.97 / 0.92 |
| `convert_cq15_cu8` | `sse4.2` 0.73 / 0.87; `avx2` 0.99 / 0.95; `neon` 0.93 / 0.93 |
| `convert_cs16` | `avx2` 0.99 / 0.93; `avx512` 0.92 / 0.72; `neon` 0.87 / 0.84 |
| `convert_cs8` | `avx2` 1.00 / 0.92; `avx512` 0.93 / 0.72; `neon` 0.85 / 0.82 |
| `convert_cu8` | `sse4.2` 0.96 / 0.74; `avx2` 0.94 / 0.72; `avx512` 0.87 / 0.56 |
| `convert_real_rs16` | `avx2` 1.01 / 0.94; `avx512` 0.94 / 0.73; `neon` 0.87 / 0.84 |
| `costas` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `costas_qam1024` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `costas_qam16` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `costas_qam256` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `costas_qam64` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `deinterleave` | `sse2` 1.00 / 0.94; `sse4.2` 1.00 / 0.94; `avx2` 0.99 / 0.98; `neon` 0.92 / 0.88 |
| `deinterleave_cq15` | `sse4.2` 0.99 / 0.63; `avx2` 0.73 / 0.86; `neon` 0.92 / 0.88 |
| `dot_complex` | `neon` 0.86 / 0.39; `simd128` 1.00 |
| `dot_cq15` | `neon` 1.00 / 1.00 |
| `equalize` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `equalize_qam1024` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `equalize_qam16` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `equalize_qam256` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `equalize_qam64` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `fir_cq15` | `sse2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 0.99 / 0.94; `simd128` 0.90 |
| `fir_decimate_cq15` | `sse2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 0.90 |
| `fir_interpolate_cq15` | `sse2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `simd128` 1.00 |
| `fir_interpolate_real_q15` | `sse2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `fm_mod` | `sse2` 0.89 / 0.77 |
| `interleave` | `sse2` 1.00 / 0.84; `sse4.2` 1.00 / 0.84; `avx2` 1.00 / 0.99; `avx512` 0.76 / 0.97; `neon` 1.00 / 0.93 |
| `interleave_cq15` | `sse2` 0.99 / 0.84; `sse4.2` 0.99 / 0.84; `avx2` 1.01 / 1.00; `avx512` 0.78 / 1.00; `neon` 1.00 / 0.93 |
| `mag` | `sse2` 0.96 / 0.99; `sse4.2` 0.96 / 0.99; `avx2` 0.84 / 0.97; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 0.83 |
| `mag_q15` | `avx512` 0.75 / 0.84; `neon` 1.00 / 0.93 |
| `mag_sq` | `sse2` 0.98 / 0.87; `sse4.2` 0.99 / 0.59; `avx2` 0.78 / 0.82; `neon` 0.85 / 0.80 |
| `mag_sq_q15` | `avx2` 0.69 / 0.80; `avx512` 0.92 / 0.71 |
| `mul` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 0.96; `avx2` 0.88 / 1.00; `neon` 0.69 / 0.90; `simd128` 1.10 |
| `mul_cq15` | `sse2` 1.00 / 1.00; `avx512` 0.86 / 0.55; `neon` 0.82 / 0.83; `simd128` 1.00 |
| `mul_real` | `sse2` 0.87 / 0.78; `sse4.2` 0.87 / 0.78; `avx2` 0.99 / 0.95 |
| `mul_real_q15` | `sse2` 1.00 / 1.00 |
| `power_db` | `avx2` 0.67 / 0.91; `simd128` 1.02 |
| `power_db_q15` | `neon` 0.96 / 0.97; `simd128` 1.03 |
| `resample_cq15` | `sse2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `simd128` 0.91 |
| `resample_real_q15` | `sse2` 1.00 / 1.00; `neon` 0.86 / 0.73 |
| `scale` | `sse4.2` 0.98 / 0.87; `avx2` 1.00 / 0.93; `avx512` 0.98 / 0.95; `neon` 0.91 / 0.90; `simd128` 0.92 |
| `scale_cq15` | `sse2` 1.00 / 1.00 |
| `scramble_c32_b1` | `neon` 0.93 / 0.92 |
| `scramble_c32_b2` | `neon` 0.77 / 0.84 |
| `scramble_cq15_b8` | `neon` 0.82 / 0.80; `simd128` 1.11 |
| `stereo_matrix` | `sse4.2` 0.85 / 0.74; `neon` 1.00 / 1.00; `simd128` 1.07 |
| `stereo_matrix_cq15` | `neon` 1.00 / 1.00; `simd128` 1.00 |
| `symbol_sync` | `sse2` 1.00 / 1.00; `sse4.2` 1.00 / 1.00; `avx2` 1.00 / 1.00; `avx512` 1.00 / 1.00; `neon` 1.00 / 1.00; `simd128` 1.00 |
| `window` | `sse2` 0.85 / 0.86; `sse4.2` 0.85 / 0.86; `avx2` 0.85 / 0.73; `avx512` 0.73 / 0.92; `neon` 0.96 / 0.95 |
| `deemphasis` | `simd128` 0.90 |
| `halfband_decimate_cq15` | `simd128` 0.90 |
| `scramble_c32_b8` | `simd128` 0.97 |
| `scramble_cq15_b2` | `simd128` 0.92 |

## Group 1: same-work references (done)

The maintainer's rule: where the judged library does less work, a library or a composition of library calls that does the same work is the judged comparison, and the other is information. Both cases have one (D2954, D2955); the code of ours is unchanged. CI run 45659904269 on e84d0603 (`measure_only`, the three entries at `sse2`, `sse4.2`, `avx2`, `avx2-runtime`, `avx512` and `neon`); the information ratios against the old comparison are those of CI run 45641571494.

`am_mod_fallback` against the CMSIS-DSP composition `arm_clip_f32` + `arm_scale_f32` + `arm_offset_f32` + the carrier product (D2954):

| Level | Ours (cycles per sample, ref. / counter) | CMSIS-DSP composition (cycles per sample, ref. / counter) | Ours / composition (ref. / counter) | Ours / liquid-dsp `ampmodem`, information (ref. / counter) | Models (ref. / counter) | SNR in dB (ours / composition) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `sse2` | 1.09 / 1.16 | 2.94 / 3.20 | 0.371 / 0.361 | 1.842 / 1.613 | skylake / znver2 | 150.8 / 150.8 | `release` (was `dev`) |
| `sse4.2` | 1.09 / 1.16 | 2.51 / 2.76 | 0.434 / 0.418 | 1.842 / 1.613 | skylake / znver2 | 150.8 / 150.8 | `release` (was `dev`) |
| `avx2` | 0.65 / 0.47 | 1.74 / 1.58 | 0.372 / 0.297 | 1.225 / 0.867 | znver3 / alderlake | 150.8 / 150.8 | `release` (was `dev`) |
| `avx512` | 0.38 / 0.30 | 1.47 / 1.25 | 0.261 / 0.243 | 0.726 / 0.560 | znver4 / sapphirerapids | 150.8 / 150.8 | `release` |
| `neon` | 2.91 / 1.52 | 5.41 / 5.19 | 0.538 / 0.294 | 2.273 / 1.478 | cortex-a72 / cortex-a76 | 150.8 / 150.8 | `release` (was `dev`) |

The composition does the clamp and the carrier product of the contract, its values are ours bit for bit; it has no NaN rule (the generic `arm_clip_f32` copies a NaN). Ours does more than the composition: the NaN rule (a comparison and a mask per vector), a NaN input gives the carrier instead of NaN. More than liquid-dsp: the clamp (the envelope stays within `carrier (1 +- depth)` beyond full scale, where liquid's `1 + m x` overmodulates and turns negative below `x = -1 / m`) and the carrier product. The accuracy gap at `neon` of group 2 (150.80 against liquid's 152.07 dB, GCC's `fmla`) is gone with the comparison: the composition is compiled without contraction like the other CMSIS-DSP references, 150.8 dB on both sides. `simd128` stays `release`: 10.54 against 49.82 instructions per sample of the composition (0.21; liquid-dsp 46.02, 0.23, information), job `count_wasm32` of CI run 45679069124 on 30d63802 (only this record taken from it; the run's other records equal the committed ones but `fft_4096`, whose comparison moved by 0.01). The `fmaxnm` form of the clamp on AArch64 was looked at again (local `llvm-mca`: 2.78 against 2.89 cycles per sample on `cortex-a72`, 1.53 against 1.52 on `cortex-a76`) and not taken.

`convert_out_cu8` and `convert_out_real_ru8` against KFR's expression of the `CU8` contract (D2955: NaN as 0 by a `select`, the `f64` formula, clamped):

| Entry | Level | Ours (cycles per sample, ref. / counter) | KFR (cycles per sample, ref. / counter) | Ours / KFR (ref. / counter) | Ours / VOLK, information (ref. / counter) | Models (ref. / counter) | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `convert_out_cu8` | `sse2` | 1.59 / 1.55 | 26.54 / 25.65 | 0.060 / 0.061 | 1.252 / 1.225 | skylake / znver2 | `release` (was `dev`) |
| `convert_out_cu8` | `sse4.2` | 1.57 / 1.52 | 18.28 / 18.27 | 0.086 / 0.083 | 1.177 / 1.094 | skylake / znver2 | `release` (was `dev`) |
| `convert_out_cu8` | `avx2` | 0.53 / 0.67 | 13.08 / 11.34 | 0.041 / 0.059 | 0.985 / 1.114 | znver3 / alderlake | `release` (was `dev`) |
| `convert_out_cu8` | `avx2-runtime` | 0.53 / 0.66 | 13.08 / 11.34 (the `avx2` record) | 0.040 / 0.059 | 0.98 / 1.10 | znver3 / alderlake | `release` (was `dev`) |
| `convert_out_cu8` | `avx512` | 0.53 / 0.49 | 9.63 / 7.90 | 0.055 / 0.062 | information | znver4 / sapphirerapids | `release` |
| `convert_out_cu8` | `neon` | 2.91 / 2.09 | 29.59 / 21.19 | 0.098 / 0.099 | 1.094 / 0.974 | cortex-a72 / cortex-a76 | `release` |
| `convert_out_real_ru8` | `sse2` | 0.81 / 0.79 | 13.28 / 12.83 | 0.061 / 0.062 | 1.260 / 1.233 | skylake / znver2 | `release` (was `dev`) |
| `convert_out_real_ru8` | `sse4.2` | 0.80 / 0.77 | 9.15 / 9.14 | 0.087 / 0.085 | 1.190 / 1.105 | skylake / znver2 | `release` (was `dev`) |
| `convert_out_real_ru8` | `avx2` | 0.27 / 0.34 | 6.55 / 5.68 | 0.042 / 0.061 | 0.992 / 1.132 | znver3 / alderlake | `release` (was `dev`) |
| `convert_out_real_ru8` | `avx2-runtime` | 0.27 / 0.34 | 6.55 / 5.68 (the `avx2` record) | 0.042 / 0.060 | 0.997 / 1.13 | znver3 / alderlake | `release` (was `dev`) |
| `convert_out_real_ru8` | `avx512` | 0.27 / 0.25 | 4.83 / 3.96 | 0.057 / 0.064 | information | znver4 / sapphirerapids | `release` |
| `convert_out_real_ru8` | `neon` | 1.47 / 1.06 | 14.82 / 10.61 | 0.099 / 0.100 | 1.102 / 0.985 | cortex-a72 / cortex-a76 | `release` (was `dev`) |

SNR 47.27 dB (`cu8`) and 47.25 dB (`ru8`) on both sides at every level. KFR's expression gives the bytes of the contract bit for bit at `sse4.2`, `avx2` and `avx512`; at `sse2` and on AArch64 its `round` rounds ties away from zero (an input on a multiple of `1 / 128`). KFR is a weak opponent here: its cast to `uint8_t` stores byte by byte, so its cost is that cast and not the work of the contract. What ours does more than VOLK, and what it costs: the NaN rule and the exact rounding, 31 against 22 vector operations per 32 floats (IO1-2), up to 1.260 / 1.233 of VOLK at `sse2`. What it gains: a NaN gives 128 (the zero of the format) instead of 255 (positive full scale); of the 4,278,190,082 non-NaN `f32` inputs VOLK's `f32` formula gives another byte for 243,270,367 (all negative inputs of magnitude at most `2^-25`, 128 instead of 127, and 735 inputs within `2^-24` of a multiple of `1 / 128`); on the measured workload the SNR is the same. The maintainer may take the VOLK ratios as the reason for a contract change of `CU8` (option (b) of the former IO1-2); without it the entries are `release` against the same-work comparison.

## Group 3b: SIMD entries no faster than their fallback (done, with open items)

Per entry a real SIMD gain or the duplicate removed (D2970 to D2973). CI run 45700747628 on 77f00faf (`measure_only`, `am_mod`, `interleave`, `deinterleave` and `stereo_matrix` with their variants at `avx2`, `avx2-runtime` and `neon`); the cases without a change were judged with `llvm-mca` of written-out loops (the corrected build of D942) against the numbers of CI run 45641571494.

| Entry | Level | Change | Ours (cycles per sample, ref. / counter) | Own fallback (cycles per sample, ref. / counter) | Ours / own fallback (ref. / counter) | Models (ref. / counter) | Status | Decision, open item |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `am_mod` | `neon` | NEON branch (`fmax` / `fmin`, `fcmeq` mask, `st2` with zero) | 2.53 / 1.40 | 2.91 / 1.52 | 0.870 / 0.917 (was 1.000 / 1.000) | cortex-a72 / cortex-a76 | `info` | D2970, IO2-10 |
| `am_mod` | `simd128` | SIMD128 branch (blocks with folded offsets) | the count of CI below | 10.54 instructions per sample | — | — | `info` | D2970 |
| `am_mod` | x86 levels | none (a written-out loop gains less than 10 % in `llvm-mca`, loses on `skylake`) | unchanged | unchanged | 1.000 / 1.000 | — | `info` | D2970, IO2-10 |
| `interleave` | `avx2` | AVX2 path removed, the fallback's loop | 0.262 / 0.514 | 0.262 / 0.514 | 1.000 / 1.000 (was 0.995 / 0.993) | znver3 / alderlake | `release` (0.976 / 0.982 of VOLK) | D2971 |
| `interleave_cq15` | `avx2` | AVX2 path removed, the fallback's loop | 0.134 / 0.134 | 0.134 / 0.134 | 1.000 / 1.000 (was 1.010 / 0.999) | znver3 / alderlake | `release` (0.020 / 0.018 of KFR), I13.1-15 closed | D2971 |
| `deinterleave` | `avx2` | AVX2 path removed, the fallback's loop | 0.263 / 0.263 | 0.262 / 0.262 | 1.004 / 1.004 (was 0.991 / 0.984) | znver3 / alderlake | `release` (0.668 / 0.662 of VOLK) | D2971 |
| `convert_cq15_cs8` | `avx2` | none (no path of its own; the generic code is the fallback's vectorized loop, bound by the stores) | 0.14 / 0.14 | 0.14 / 0.14 | 0.992 / 0.986 | znver3 / alderlake | `release` | D2971 |
| `stereo_matrix` | `neon` | NEON path removed, the fallback | 1.899 / 1.392 | 1.899 / 1.392 | 1.000 / 1.000 (was 1.001 / 1.001) | cortex-a72 / cortex-a76 | `release` (0.331 / 0.266 of VOLK) | D2972 |
| `stereo_matrix_cq15` | `neon` | NEON path removed, the fallback | 0.765 / 0.763 | 0.765 / 0.763 | 1.000 / 1.000 (was 1.002 / 1.002) | cortex-a72 / cortex-a76 | `info` | D2972 |
| `mag` | `avx512`, `neon` | none (the vector square root is the bound) | 0.64 / 0.76; 8.58 / 1.77 | 0.64 / 0.76; 8.56 / 1.77 | 0.999 / 0.999; 1.002 / 0.997 | znver4 / sapphirerapids; cortex-a72 / cortex-a76 | `release` | D2973, IO2-11 |
| `mul` | `sse2` | none (port 5 of `skylake` is the bound) | 1.53 / 1.59 | 1.53 / 1.59 | 1.001 / 1.001 | skylake / znver2 | `release` | D2973, IO2-12 |
| `dot_cq15` | `neon` | none (the compiled fallback, since M13) | 0.90 / 0.90 | 0.90 / 0.90 | 0.999 / 1.000 | cortex-a72 / cortex-a76 | `info` | D2973, IO2-13 |
| `equalize` (five entries) | every level | none (the inner parts are vectorized by LLVM already; the chain per symbol is the bound) | as in the target table | as in the target table | 1.000 / 1.000 | — | `info` | D2973, IO2-14 |

Attempts in `llvm-mca` (cycles per sample, reference / counter model, written-out loops): `am_mod` x86 unrolled 1.159 / 1.086 against the compiled 1.059 / 1.130 (`sse2`) and 0.564 / 0.440 against 0.628 / 0.443 (`avx2`); `am_mod` NEON with `zip1` / `zip2` and `stp` 2.75 / 1.63, with `st2` 2.38 / 1.38 against the compiled 2.88 / 1.50; `stereo_matrix` NEON without `ld2` / `st2` 2.38 / 1.75 against 2.00 / 1.38; `mag` Newton form 0.629 / 0.503 against 0.627 / 0.753 (`avx512`) and 5.83 / 3.27 against 8.51 / 1.76 (`neon`); `mul` SSE2 without SSE3 1.68 / 2.01 against 1.50 / 1.57. The `avx2-runtime` records of the removed AVX2 paths: `interleave` 0.266 / 0.392, `interleave_cq15` 0.139 / 0.137, `deinterleave` 0.266 / 0.265 (information against the fixed `avx2` build: 1.016 / 0.762, 1.032 / 1.023, 1.011 / 1.007).

## Status

- 2026-11-25: planned; group 0 inventory above; the maintainer's decisions recorded (D2950 to D2953).
- 2026-11-25: full CI measurement of every entry on every level on 05ff2d25 (CI run 45641571494, 1836 records, 10 of them more than 1 % from the committed numbers, none of them a status) and the `simd128` counts of its job `count_wasm32` (unchanged); records merged, `cargo xtask check cycles` passes; target list above. No status changes; next: group 1 and group 2 (`oscillate`).
- 2026-11-25: group 1 done (D2954, D2955; CI run 45659904269 on e84d0603): `am_mod_fallback` judged against the CMSIS-DSP composition of the same work, `release` at every level (0.243 to 0.538); `convert_out_cu8` and `convert_out_real_ru8` judged against KFR's expression of the `CU8` contract, `release` at every x86-64 level and `neon` (0.040 to 0.100); liquid-dsp and VOLK information; IO1-2 and IO1-3 closed. The leftover job `test_m55_qemu` of CI run 45641571494 (the base commit, its measurement collected) was cancelled because the concurrency group held this run behind it.
- 2026-11-25: the `simd128` count of `am_mod_fallback` against the CMSIS-DSP composition (CI run 45679069124, `count_wasm32`): 10.54 against 49.82 instructions per sample (0.21), `release`.
- 2026-11-25: group 3b done with open items (D2970 to D2973; CI run 45700747628 on 77f00faf): `am_mod` `C32` has NEON and SIMD128 branches (`neon` 0.870 / 0.917 of the fallback, IO2-10); the AVX2 paths of `interleave`, `interleave_cq15` and `deinterleave` and the NEON paths of `stereo_matrix` and `stereo_matrix_cq15` are removed, the fallback's loops run (1.000 / 1.000, I13.1-15 closed); `mag`, `mul` at `sse2`, `dot_cq15` at `neon` and `equalize` keep their code with the reasons and attempts in IO2-11 to IO2-14. IO2-1 to IO2-9 and D2956 to D2969 stay free for groups 2 and 3a.
