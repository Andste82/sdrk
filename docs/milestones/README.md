# SDRk milestones

> Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 · Snapshot: [`docs/PLAN.md`](../PLAN.md)

The plan explains **why** and in which phases; the milestones say **exactly what** to build, in which order, and how to prove it works. Each milestone is self-contained: it repeats the ground rules, defines its public APIs, numeric rules, golden test vectors (with seeds and the exact Python recipe), tests, acceptance commands and a deliverables checklist. A capable but less experienced implementer — human or AI — should be able to finish a milestone from its file alone plus the code delivered by the earlier milestones.

## Order and dependencies

Up to M25.1 the milestones are strictly sequential: each depends on all previous ones, and each must build and pass all tests on its own before the next one starts. **From M26 on, a milestone depends only on the milestones named in its *Depends on* line** (rule R1.1 of those files): milestones that do not depend on each other may be implemented in parallel, each on its own branch, and the second one to merge rebases onto the first. State on `main`: M00 to M15, M19 to M25, M24.1 and M25.1 (release 0.1.0) are delivered; M26 to M44 and the backends M16 to M18 are planned. M16 to M18 come after M29 (the backends then also cover the primitive blocks; their kernels are unchanged by M26 to M29). From M37 on, rule R2 includes the crate `sdrk-proto` (`crates/proto`).

**Block names:** M24.1 fixes the naming scheme of block kinds (`docs/blocks.md`, section "Names") and renames four kinds: `resampler` → `resample`, `halfband` → `halfband_decimate`, `real_to_complex` → `real_to_iq`, `fm_stereo` → `fm_stereo_demod` (a hard break: the old names are not accepted, maintainer decision of 2026-11-01). The milestone files before M24.1 are records and use the old names; every later milestone uses the scheme and the table of reserved names.

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
| M24.1 | [M24.1-block-naming.md](M24.1-block-naming.md) | 9 | Block names: naming scheme in `docs/blocks.md`, kinds `resample`, `halfband_decimate`, `real_to_iq`, `fm_stereo_demod` (no aliases for the old names), name check in `cargo xtask check`, reserved names for the planned blocks; outputs unchanged |
| M25 | [M25-fm-transmit-audio.md](M25-fm-transmit-audio.md) | 9 | FM transmit audio: kernel `preemph_limit`, design `preemphasis_coefs`, block `preemph_limit`, pre-emphasis and limiter in `fm_transmitter`, flow `nfm_transmitter`, de-emphasis in `nfm_receiver`, loop tests TX → RX |
| M25.1 | [M25.1-release-0.1.md](M25.1-release-0.1.md) | 9 | Release 0.1.0: release condition with accepted waivers, `cargo xtask release-check`, version 0.1.0, CHANGELOG section, release notes; the tag is the maintainer's step |
| M26 | [M26-block-ports.md](M26-block-ports.md) | 9 | Block ports: `StreamElem`, `Ports`, `Two<A, B>`, `Frame`; `FrameBlock` removed (`spectrum` becomes a `Block` with frame output) |
| M27 | [M27-chain.md](M27-chain.md) | 9 | `Chain<A, B, S>` and `chain!`: typed, heap-free composition of blocks; parameter paths; target test on Cortex-M4 |
| M28 | [M28-clock-drift.md](M28-clock-drift.md) | 9 | `DriftController`: PI control of the tunable `resample` from a buffer fill; `PipelineDrift`; testkit `TwoClockSim` |
| M29 | [M29-primitive-blocks.md](M29-primitive-blocks.md) | 9 | Primitive blocks `quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`, `mag`, `delay` with kinds |
| M30 | [M30-two-port-blocks.md](M30-two-port-blocks.md) | 9 | Kernels `add`, `mul`, `oscillate`, `interleave`; blocks `add`, `mul`, `make_complex`, `to_complex`, `real_part`, `imag_part`, `nco`, `stereo_pll`, `stereo_matrix` |
| M31 | [M31-composite-tools.md](M31-composite-tools.md) | 9 | `Chain` with `Two` intermediates, `Fork` with delay alignment, `ParamMap` |
| M32 | [M32-composite-plans.md](M32-composite-plans.md) | 9 | `sdrk_blocks::plans`: the six specific blocks as compositions of primitives, equal bit for bit; real `dc_block` |
| M33 | [M33-noise-reduction.md](M33-noise-reduction.md) | 9 | Noise blanker and automatic notch (kernels, blocks, kinds), flow `hf_am_receiver` |
| M34 | [M34-multi-input.md](M34-multi-input.md) | 9 | Two-input nodes in the pipeline: edge `port`, lockstep runner, `rate_mismatch`/`clock_domain`/`misaligned_inputs`, automatic alignment, proof flow `fm_stereo_receiver_primitives` |
| M35 | [M35-am-ssb-tx.md](M35-am-ssb-tx.md) | 9 | AM and SSB transmit: kernel and block `am_mod`, block `ssb_mod`, flows `am_transmitter`, `ssb_transmitter`, loop tests |
| M36 | [M36-fm-stereo-tx.md](M36-fm-stereo-tx.md) | 9 | FM stereo transmit: stereo `preemph_limit` with linked gain, `stereo_encode`/`fm_stereo_mod`, flow `fm_stereo_transmitter` |
| M37 | [M37-packet-ports.md](M37-packet-ports.md) | 10 | Packet ports, `Clocked<S, Packets>`, crate `sdrk-proto` (packet, CRCs, test framing), packet inputs/outputs, JSON Lines |
| M38 | [M38-symbol-sync.md](M38-symbol-sync.md) | 10 | `symbol_sync` (Gardner, Mueller–Müller), `slice`, PRBS, testkit channel model and BER meter |
| M39 | [M39-fsk.md](M39-fsk.md) | 10 | FSK/GFSK RX+TX (`fsk_demod`, `fsk_mod`), `afc` with ppm report, PN9 whitening, GFSK packet flows |
| M39.1 | [M39.1-afsk-ook-adsb.md](M39.1-afsk-ook-adsb.md) | 10 | AFSK with NRZI/HDLC/AX.25 (APRS RX+TX), OOK RX+TX, ADS-B decoder |
| M40 | [M40-psk.md](M40-psk.md) | 10 | BPSK/QPSK RX+TX, coherent and differential, `costas`, RRC design, QPSK flows |
| M40.1 | [M40.1-psk-variants.md](M40.1-psk-variants.md) | 10 | 8PSK, π/4-DQPSK, OQPSK |
| M41 | [M41-rds-rx.md](M41-rds-rx.md) | 10 | RDS receive: `rds_demod`, `rds_decode`, `sdrk_proto::rds`, flow `fm_rds_receiver`, E2E-RDS |
| M42 | [M42-rds-tx.md](M42-rds-tx.md) | 10 | RDS transmit: group carousel, `rds_encode` (`Clocked`), `fm_stereo_rds_mod`, flow `fm_stereo_rds_transmitter` |
| M43 | [M43-equalizer.md](M43-equalizer.md) | 10 | Adaptive equalizer `equalize` (CMA, decision-directed), testkit multipath |
| M44 | [M44-qam.md](M44-qam.md) | 10 | 16-QAM and 64-QAM RX+TX |
| — | [BACKLOG.md](BACKLOG.md) | 7–10, later | Remaining backends, optional C/Python bindings, extensions, stabilization |

```text
M00 → M01 → M02 → M03 → M04 → M05 → M06 → M07 → M08     (phases 0–2: foundation and kernels)
                                                   ↓
                         M12 ← M11 ← M10 ← M09 ←───┘     (phase 3: blocks, pipeline, I/O)
                          ↓
                         M13 → M14 → M15                 (phases 4–6: speed, WASM, emulators)
                                      ↓
                                     M19 → M20 → M21 → M22 → M23 → M24 → M24.1 → M25 → M25.1    (phase 9, strictly sequential)
```

From M26 on, the dependencies (the *Depends on* line of each file; dependencies of dependencies are implied):

| Milestone | Depends on | Can run in parallel with |
| --- | --- | --- |
| M26 Block ports | M25.1 | M28 |
| M27 Chain | M26 | M28, M29, M30, M33, M34 |
| M28 Clock drift | M25.1 | everything from M26 on |
| M29 Primitive blocks | M26 | M27, M28, M30, M33 |
| M30 Two-port blocks | M26 | M27, M28, M29, M33 |
| M31 Composite tools | M27, M29, M30 | M33, M34, M35, M36 |
| M32 Composite plans | M31 | M33–M36 |
| M33 Noise reduction | M26 | M27–M32, M34–M36 |
| M34 Multi-input pipeline | M26, M29, M30 | M27, M28, M31–M33, M35, M36 |
| M35 AM/SSB transmit | M29, M30 | M31–M34, M36 |
| M36 FM stereo transmit | M30 | M31–M35 |
| M37 Packet ports, `sdrk-proto` | M34 | M35, M36 |
| M38 Symbol sync, BER | M29, M37 | M35, M36 |
| M39 FSK, AFC | M38 | M40 |
| M39.1 AFSK/APRS, OOK, ADS-B | M39 | M40, M40.1, M41 |
| M40 BPSK/QPSK | M38 | M39, M39.1 |
| M40.1 8PSK, π/4-DQPSK, OQPSK | M40 | M39.1, M41 |
| M41 RDS receive | M40 | M39.1, M40.1, M43, M44 |
| M42 RDS transmit | M41, M36, M37 | M43, M44 |
| M43 Equalizer | M40.1 | M41, M42 |
| M44 QAM | M43 | M41, M42 |
| M16 → M17 → M18 Backends | M15 (scheduled after M29, so the primitive blocks are covered) | everything after M29 |

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
| 436–437 | M20 (the audio-filter cases of `am_demod`, open item M20-5) |
| 438–443 | M24 |
| 444–450 | M25 |
| 451 | M21 (the loud decimated `CQ15` case `ssb_demod_usb_cq15_dec4_loud`, review finding) |
| 452–453 | M23 (the saturation-corner cases of the `Q15` halfband kernels, review finding) |
| 454 | M24 (the additional case `stereo_pll/disable_enable_f32_16384`, D113) |
| 455–461 | M29 |
| 462 | M28 |
| 463–468 | M30 |
| 469 | M31 |
| 470 | M32 |
| 471–477 | M33 |
| 478 | M34 |
| 479 | M37 |
| 480–484 | M35 |
| 485–489 | M36 |
| 490–499 | M38 |
| 500–519 | M17 (RVV stress and digest inputs; no golden vectors) |
| 520–529 | M18 (arm-dsp stress inputs; no golden vectors) |
| 530–539 | M39 |
| 540–549 | M39.1 |
| 550–559 | M40 |
| 560–569 | M40.1 |
| 570–579 | M41 |
| 580–589 | M42 |
| 590–599 | M43 |
| 600–619 | M44 |
| 620–999 | free, for backlog milestones |
| 1000+ | benchmark inputs (M13, M16) — not golden vectors |

## Files in this directory

- `M00-…` to `M44-…`: the milestones (each contains the full ground rules; from M26 on with the changed rule R1.1, from M37 on with `sdrk-proto` in R2).
- `BACKLOG.md`: work after M18 and M44.
