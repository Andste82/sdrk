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

TARGETS

## Status

- 2026-11-25: planned; group 0 inventory above; the maintainer's decisions recorded (D2950 to D2953). Full CI measurement of every entry on every level and the `simd128` counts on 05ff2d25 (CI run 45641571494).
