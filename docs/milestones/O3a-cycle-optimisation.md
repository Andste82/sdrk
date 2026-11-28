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

RESULTS_SECTION
