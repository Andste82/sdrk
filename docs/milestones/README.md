# SDRk milestones

> Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 · Snapshot: [`docs/PLAN.md`](../PLAN.md)

The plan explains **why** and in which phases; the milestones say **exactly what** to build, in which order, and how to prove it works. Each milestone is self-contained: it repeats the ground rules, defines its public APIs, numeric rules, golden test vectors (with seeds and the exact Python recipe), tests, acceptance commands and a deliverables checklist. A capable but less experienced implementer — human or AI — should be able to finish a milestone from its file alone plus the code delivered by the earlier milestones.

## Order and dependencies

Milestones are strictly sequential: each depends on all previous ones, and each must build and pass all tests on its own before the next one starts. State on `main`: M00 to M15 and M19 are delivered; M16 to M18 and M20 to M25 are planned (the receive series M19 to M23, M24 and M25 use nothing from M16 to M18, see the *Order* section of M19).

| # | File | Plan phase | Delivers |
| --- | --- | --- | --- |
| M00 | [M00-workspace-foundation.md](M00-workspace-foundation.md) | 0 | Workspace, layer folders, `xtask` checks (layers, heap, global state, docs, bare-metal), lints, `cargo deny`, CI on Linux x86_64/arm64 and Windows x86_64, docs skeleton |
| M01 | [M01-device-foundations.md](M01-device-foundations.md) | 1 | `sdrk-device`: error scheme, `SampleFormat` + device constants, time types, stream traits and block metadata |
| M02 | [M02-core-foundations.md](M02-core-foundations.md) | 1 | `sdrk-core`: sample types, Q15 arithmetic, timing, kernel pattern, facade `Sdrk`, `delegate!`, example kernel `scale` |
| M03 | [M03-testkit-foundations.md](M03-testkit-foundations.md) | 1 | `sdrk-testkit`: PCG32, generators, metrics, arena, encode oracle, `VirtualRx`, `no_std` suite and protocol |
| M04 | [M04-golden-vectors.md](M04-golden-vectors.md) | 1 | Golden-vector pipeline (Python 3.12 + NumPy/SciPy), file format, Q15 model, Rust loader, first cases |
| M05 | [M05-first-kernels.md](M05-first-kernels.md) | 1 | `convert` (all formats), `rotate`, `fir` |
| M06 | [M06-receive-kernels.md](M06-receive-kernels.md) | 2 | `fir_decimate`, `resample`, `quad_demod`, `deemphasis`, `dc_block`, windows, `mag_sq`, `power_db` |
| M07 | [M07-fft.md](M07-fft.md) | 2 | Own FFT: f32/Q15, fixed and block-floating-point scaling, real FFT and its inverse, static twiddles |
| M08 | [M08-transmit-and-fused.md](M08-transmit-and-fused.md) | 2 | `convert_out`, `fir_interpolate`, `fm_mod`, fused `ddc`/`duc`/`fm_demod`, meters, FM end-to-end chains |
| M09 | [M09-blocks.md](M09-blocks.md) | 3 | `sdrk-blocks`: filter design, storage, block interface with descriptions, 11 blocks, kernel `dot` |
| M10 | [M10-pipeline-core.md](M10-pipeline-core.md) | 3 | `sdrk-pipeline`: JSON model, registry, schema, validation, build, push runner, no allocation after build |
| M11 | [M11-pipeline-time-events-tx.md](M11-pipeline-time-events-tx.md) | 3 | Time base, events, timed parameters, pull mode/bursts, groups, YAML, wire format, two-thread probe |
| M12 | [M12-io-device-loopback-examples.md](M12-io-device-loopback-examples.md) | 3 | `sdrk-io` (SigMF/WAV/raw), device capabilities + async traits, conformance, loopback, CLI examples, system tests |
| M13 | [M13-simd-backend.md](M13-simd-backend.md) | 4 | `sdrk-backend-simd`, SIMD FFT, instruction counting vs VOLK/rustfft, CI for AVX-512/AArch64/WASM; the tag v0.1.0 is specified but not made (open item M13-4) |
| M14 | [M14-wasm-bindings.md](M14-wasm-bindings.md) | 5 | WASM bindings, SIMD/scalar variants, npm package, Node/Chrome/Firefox tests |
| M15 | [M15-emulator-targets.md](M15-emulator-targets.md) | 6 | Suite bare-metal in QEMU (Cortex-M4/M55, RISC-V), esp-emulator (P4), Espressif QEMU (S3) |
| M16 | [M16-esp32-p4-pie-backend.md](M16-esp32-p4-pie-backend.md) | 7 | ESP32-P4 PIE backend, static instruction counting vs ESP-DSP |
| M17 | [M17-rvv-backend.md](M17-rvv-backend.md) | 7 | `sdrk-backend-rvv` (RISC-V Vector 1.0, vector-length agnostic), VLEN matrix in QEMU (128–1024), digest test for VLEN independence, `qemu-riscv64-rvv` target, instruction counts vs fallback |
| M18 | [M18-arm-dsp-backend.md](M18-arm-dsp-backend.md) | 7 | `sdrk-backend-arm-dsp` (Cortex-M4/M7/M33 DSP extension, bit-exact CQ15 with 64-bit accumulators), M7/M33 targets, instruction counts vs CMSIS-DSP |
| M19 | [M19-nfm-squelch.md](M19-nfm-squelch.md) | 9 | NFM receive: kernel and block `squelch`, flow `nfm_receiver`, E2E-NFM; introduces the exact float32 golden model |
| M20 | [M20-am-receive.md](M20-am-receive.md) | 9 | AM receive: kernels `mag` (SIMD), `dc_block_real`, `agc`; blocks `agc`, `am_demod`; flows `am_receiver`, `airband_receiver`; E2E-AM |
| M21 | [M21-ssb-receive.md](M21-ssb-receive.md) | 9 | SSB receive: block `ssb_demod` from existing kernels, flow `ssb_receiver`, E2E-SSB with sideband rejection |
| M22 | [M22-audio-filters.md](M22-audio-filters.md) | 9 | Audio filters: kernels `biquad`/`biquad_real`, RBJ and Butterworth design, block `biquad`, CTCSS removal and audio low-pass in the flows |
| M23 | [M23-real-sampling-input.md](M23-real-sampling-input.md) | 9 | Real-sampling devices: `halfband_decimate`, `real_to_complex` (SIMD), real formats in `convert`, blocks `halfband`, `real_to_complex`, E2E-REAL |
| M24 | [M24-fm-stereo.md](M24-fm-stereo.md) | 9 | FM stereo: kernels `stereo_pll` (pilot PLL) and `stereo_matrix` (SIMD), block `fm_stereo`, stereo audio as complex samples, flow `fm_stereo_receiver`, stereo WAV, E2E-STEREO |
| M25 | [M25-fm-transmit-audio.md](M25-fm-transmit-audio.md) | 9 | FM transmit audio: kernel `preemph_limit`, design `preemphasis_coefs`, block `fm_tx_audio`, pre-emphasis and limiter in `fm_transmitter`, flow `nfm_transmitter`, de-emphasis in `nfm_receiver`, loop tests TX → RX |
| — | [BACKLOG.md](BACKLOG.md) | 7–9, later | Remaining backends, optional C/Python bindings, extensions, stabilization |

```text
M00 → M01 → M02 → M03 → M04 → M05 → M06 → M07 → M08     (phases 0–2: foundation and kernels)
                                                   ↓
                         M12 ← M11 ← M10 ← M09 ←───┘     (phase 3: blocks, pipeline, I/O)
                          ↓
                         M13 → M14 → M15 → M16 → M17 → M18 → BACKLOG                (phases 4–7: speed, WASM, embedded)
                                      ↓
                                     M19 → M20 → M21 → M22 → M23 → M24 → M25 → BACKLOG    (phase 9: NFM, AM, SSB, audio filters, real-sampling input, FM stereo, FM transmit audio)
```

## How to work on a milestone

1. Read the **ground rules** section of the milestone completely (it is identical in every file). They define the layer rules, `no_std`/no-heap rules, interface rules and the one error-handling scheme for the whole workspace.
2. Read the whole milestone, then the parts of earlier milestones it refers to (APIs you will call).
3. Implement in the order of the *Specification* sections. Public names, signatures and numeric rules are normative; internal structure is yours to choose.
4. Generate golden vectors with the Python recipe given in the milestone (`tests/golden/generate.py --kernel <name>`), commit the generated files together with the code, and run `--check`.
5. Run every command of R1.2 and of the milestone's *Acceptance* section on your host; CI checks the three host platforms.
6. Tick the deliverables checklist, update `CHANGELOG.md` (Unreleased section) and `docs/decisions-log.md` (every ambiguity you resolved), and update the docs named in the milestone.
7. One milestone = one branch and one pull request with the milestone number in the title (`M05: convert, rotate, fir`).

If a milestone contradicts an earlier one or the plan, stop and resolve it explicitly (decisions log + update of the affected files) instead of guessing.

## Seed ranges for golden vectors

Every golden case has a fixed seed; ranges keep cases independent so that adding a case never changes another one.

| Range | Used by |
| --- | --- |
| 0 | deterministic cases without randomness (filter design tables, ISO 8601 table, I/O headers) |
| 1–5 | M04 (`q15_ops`, `scale`) |
| 20–39 | M05 (`rotate` 20–24, `fir` 30–34) |
| 40–69 | M06 (40–62) |
| 70–89 | M07 (70–87) |
| 90–99 | M08 kernels (`fir_interpolate`, `fm_mod`) |
| 100–199 | M05 `convert` (100 + position in the format table) |
| 200–299 | M08 `convert_out` (200 + i from C32, 220 + i from CQ15) |
| 300–399 | M09 (`dot`, blocks) |
| 400–409 | M16 (embedded FFT cases) |
| 410–412 | M19 |
| 413–420 | M20 |
| 421–423 | M21 |
| 424–429 | M22 |
| 430–435 | M23 |
| 436–441 | M24 |
| 442–448 | M25 |
| 449–499 | free, for backlog milestones |
| 500–519 | M17 (RVV stress and digest inputs; no golden vectors) |
| 520–529 | M18 (arm-dsp stress inputs; no golden vectors) |
| 530–999 | free, for backlog milestones |
| 1000+ | benchmark inputs (M13, M16) — not golden vectors |

## Files in this directory

- `M00-…` to `M25-…`: the milestones (each contains the full ground rules).
- `BACKLOG.md`: work after M18 and M25.
