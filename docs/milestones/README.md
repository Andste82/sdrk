# SDRk milestones

> Plan (living document): https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1 · Snapshot: [`docs/PLAN.md`](../PLAN.md)

The plan explains **why** and in which phases; the milestones say **exactly what** to build, in which order, and how to prove it works. Each milestone is self-contained: it repeats the ground rules, defines its public APIs, numeric rules, golden test vectors (with seeds and the exact Python recipe), tests, acceptance commands and a deliverables checklist. A capable but less experienced implementer — human or AI — should be able to finish a milestone from its file alone plus the code delivered by the earlier milestones.

## Order and dependencies

Up to M25.1 the milestones are strictly sequential: each depends on all previous ones, and each must build and pass all tests on its own before the next one starts. **From M26 on, a milestone depends only on the milestones named in its *Depends on* line** (rule R1.1 of those files): milestones that do not depend on each other may be implemented in parallel, each on its own branch, and the second one to merge rebases onto the first. State on `main`: M00 to M15, M19 to M25, M24.1 and M25.1 (release 0.1.0) are delivered; M26 to M46 and the backends M16 to M18 are planned. M16 to M18 come after M29 (the backends then also cover the primitive blocks; their kernels are unchanged by M26 to M29). From M37 on, rule R2 includes the crate `sdrk-proto` (`crates/proto`).

**Speed measure (M13.1):** the specs of implemented milestones are the record of what was done and are not edited; their instruction counts (`cargo xtask count` for `sse2`, `sse4.2`, `avx2`, `avx2-runtime`, `avx512` and `neon`, instruction ratios as a gate) are replaced by [M13.1](M13.1-cycles-only.md): simulated cycles per sample on the reference and counter model of each level (`cargo measure`, `cargo xtask check cycles`) are the only speed measure on the out-of-order levels, and instruction counts stand only for `simd128` and the in-order cores (Cortex-M4 today). The specs of planned milestones (M16 to M18, M51 to M56) are written for that rule.

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
| M13 | [M13-simd-backend.md](M13-simd-backend.md) | 4 | `sdrk-backend-simd`, SIMD FFT, instruction counting vs VOLK/rustfft, CI for AVX-512/AArch64/WASM; the release 0.1.0 is made by M25.1 (M13-4 closed, D202); open item M13-7, M13-10 (c) |
| M13.1 | [M13.1-cycles-only.md](M13.1-cycles-only.md) | 4 | Implemented (D2700 to D2749 and D2760 to D2765, PR #196, merged 2026-11-24; open items I13.1-10 (`simd128`) and I13.1-12). Cycles per sample as the only speed measure on the out-of-order levels: the release gate of `benches/limits.json` in cycles, the instruction counts removed except `simd128` and the in-order cores (Cortex-M4), committed cycle results, a nightly `cargo measure` job in CI instead of the count jobs (D2700 to D2749) |
| O1 | [O1-cycle-optimisation.md](O1-cycle-optimisation.md) | 4 | Done (planned 2026-11-24, branch `opt-round-cycles-r`, PR #204, merged 2026-11-25; D2800 to D2899; open items IO1-7, IO1-8). Cycle optimisation round: every `dev` entry above 1.10 of its library in cycles (I13.1-7, I13.1-8, I13.1-10, I24-3, with the entries above 1.50 of D2779), the run-time dispatch of the `Q15` FIR entries (I13.1-9) and SIMD128 references built from VOLK's SSE kernels |
| O2 | [O2-cycle-optimisation.md](O2-cycle-optimisation.md) | 4 | Done (planned 2026-11-25, branch `opt-round-2`, PR #206, merged 2026-11-27; O2b PR #207 and O2c PR #209, merged 2026-11-28; D2950 to D2999, D3000 to D3021, open items IO2-17, IO2-19, IO2-27 to IO2-29). O2b (2026-11-27, branch `opt-round-2b`, D3000 to D3049): the contract trade, `CU8` against VOLK, `mul` at `avx512`, the x86 `stereo_matrix` fallback, the Taylor step of the `C32` oscillator (150.56 dB, I13.1-13 and IO2-1 closed), group 3a: the grouped `Q15` / `CQ15` FIR branches on every level (the D65 switch removed but for `fir_decimate` `CQ15` on AArch64) and the `Q15` kernels at `sse2` done, the committed Cortex-M4 counts those of the nightly job (D3009, I48-2 closed), the review fixes (D3010 to D3012); D3000 to D3012, IO2-19 to IO2-27 open (the `CQ15` FFT at `neon`, IO1-5 to IO1-7 and `channelize_fallback` not done); issue #203 updated. O2c (2026-11-28, branch `opt-round-2c`, D3013 to D3021): VOLK's AVX2 protokernel judged for the `CU8` outputs at `avx512`, the two `Q15` biquads at `sse2` and `fir_decimate` `CQ15` on AArch64 without a switch, the x86 `core::arch` module of the `Q15` fallbacks (IO1-5, IO1-6 closed), the accepted explicit forms (the paths of D2971 and D2972 back), every gap above 1.10 with its reason (`slower_because`); IO2-17, IO2-19, IO2-27 to IO2-29 open. Cycle optimisation round 2: the inventory of SIMD paths that run the fallback, same-work references (`am_mod`, the `CU8` output), accuracy (`oscillate` first), grouped `Q15`/`CQ15` FIR SIMD branches instead of the D65 switch, SSE2 forms of the `Q15` kernels, SIMD entries no faster than their fallback, the rest above 1.10 of the library, report-only checks; issue #203 updated at the end |
| O3a | [O3a-cycle-optimisation.md](O3a-cycle-optimisation.md) | 4 | Ready for review (2026-11-28, branch `opt-round-3a-2`; D3100 to D3149, open items IO3-n). Cycle optimisation round 3a: the SIMD paths slower than their fallback (IO2-28: the planar `CQ15` FIR progressions), the gaps above 1.10 without a reason (IO1-7: `dot_cq15`, `mag_sq`, `deinterleave_cq15` and the interpolator fallbacks), the `simd128` fallbacks (IO2-29), `fft_cq15_1024_fallback` judged against CMSIS-DSP (IO3-1) |
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
| M32 | [M32-composite-plans.md](M32-composite-plans.md) | 9 | reference tests of six blocks (`fir`, `biquad`, `real_to_iq`, `convert`, `convert_out`, `fm_mod`: block equals kernel); real `dc_block`; the plans moved to the backlog (D358) |
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
| M44 | [M44-qam.md](M44-qam.md) | 10 | **Delivered** (2026-11-09, branch `m44`). 16-QAM and 64-QAM RX+TX: shared Gray constellations `qam16`/`qam64`, `slice`, `equalize` and `costas` (parameter `constellation`, decision-directed QAM loop) extensions, kinds `qam_mod` and `qam_demod`, `frame_decode` `rotations` and `bits_per_symbol`, flows `qam16_transmitter` and `qam16_receiver` |
| M44.1 | [M44.1-qam-building-blocks.md](M44.1-qam-building-blocks.md) | 10 | **Implemented** (2026-11-16, branch `m44-1-qam-building-blocks`; I44.1-2 to I44.1-6 decided or closed, D1587 to D1590, D1823, D1826, D3161). QAM in building blocks: `symbol_map` / `symbol_demap` (BPSK to 64-QAM; `slice` renamed), `iq_offset`, `qam_acquire`, the supervisor `LockWatch` and a control interface of `equalize`, `costas`, `symbol_sync`; `qam_mod`, `psk_mod` and `qam_demod` as composite blocks (bit-identical first, then the recovery rules: no error state lasts), composites `qam_frame_encode` / `qam_frame_decode` with a frame-level watch; the parts OFDM reuses |
| M44.1.1 | [M44.1.1-parameter-names.md](M44.1.1-parameter-names.md) | 10 | **Implemented** (2026-11-17, branch `m44-1-1-parameter-names`; I44.1.1-1 to I44.1.1-3 and I44.1-5 decided, D1823 to D1826; I44.1.1-4 closed by D3167 for the maintainer). One name per parameter: `constellation` instead of `order` in the QAM and PSK blocks, the pulse filter as the building blocks `pulse_shape` / `pulse_match`, every QAM and PSK composite forwards through `ParamMap` with no conversion |
| M44.2 | [M44.2-qam-any-order.md](M44.2-qam-any-order.md) | 10 | Implemented (2026-11-19, branch `m44-2-qam-any-order-r`; I44.2-7 to I44.2-11 decided, D1540 to D1544; test 7 is the acceptance test of the optimization round after M44.2; I44.2-12 accepted, D1547). QAM of any square order (4, 16, 64, 256, 1024): general constellation tables, soft bits (`qam_soft_bits` with a general SIMD path), `costas`/`equalize` decisions and lock metric, frequency acquisition and `frame_decode` rotations per order; 16/64 fast paths kept; flows `qam256_transmitter`/`qam256_receiver`; generalises the building blocks of M44.1 |
| M44.3 | [M44.3-equalizer-switch.md](M44.3-equalizer-switch.md) | 10 | Done (2026-11-21, branch `m44-3-equalizer-switch`; the maintainer's decisions D2421 to D2424 on I44.3-1 to I44.3-4: the measured switch of §1 taken back out and deferred to M44.4, the restart of a false state off and decided in M44.4 by an SNR estimate, `qam16_receiver` at the unit input power; test 7 passes at 999 / 1000 / 999; the cycles equal M44.2). The equalizer switches to decision mode on a measured criterion of the constellation instead of a count, a false state without lock is restarted; acceptance: the 256-QAM flow loop at 990 of 1000 packets |
| M44.4 | [M44.4-snr-restart-dense-blind-start.md](M44.4-snr-restart-dense-blind-start.md) | 10 | Implemented, §2 deferred by decision (2026-11-22, branch `m44-4-snr-restart`; the maintainer's decisions D2525, D2526 and D2528 to D2538 on I44.4-1 to I44.4-7, no open item left). §1 done: the block and kind `snr_estimate` (decision estimate with a least-squares gain and the inverse of its exact reading over white noise; 2.85 cycles per symbol at `avx2` against 50.03 of GNU Radio's M2M4). §2 deferred (D2525, I44.4-1 option (a)): on the input of the decision the estimate reads the SNR the equalizer reached, so it cannot tell a false state from a weak signal; after the multi-modulus start no 256-QAM false state is left, and the restart stays off. §3 done: the multi-modulus start of 256-QAM and 1024-QAM (1024-QAM through `mild` at Eb/N0 29 dB 15 of 16 lock, M44.3 none; 256-QAM every receiver no later than M44.3); qpsk, 16-QAM and 64-QAM bit-identical; every entry within 0.11 % of main. The information test of M44.3 for 1024-QAM judges a reported lock by the decision error, `mse / h²` below 0.5 (D2526, I44.4-2 option (c)) |
| M45 | [M45-channelizer.md](M45-channelizer.md) | 10 | **Delivered** (2026-11-08, branch `m45-channelizer`). Channelizer: kernel `channelize` (polyphase filter bank, prototype with passband and stopband edges), port type `Lanes<T>` (M channels through one output), kinds `channelize` and `channel_select`, flow `pmr446_monitor` (eight PMR446 channels at once) |
| M45.1 | [M45.1-ism-band-receiver.md](M45.1-ism-band-receiver.md) | 10 | **Delivered** (2026-11-08; I45.1-1 to I45.1-7 closed on 2026-11-09, D1021 to D1034; I45.1-8 closed by D3166). ISM band receiver: `band_pulse_detect` on the lanes of `channelize` (pulse detection in every channel, OOK and FSK, merged across channels), pulse trains as packets, flows `ism433_pulses`, `ism868_pulses` |
| M46 | [M46-ism-sensors.md](M46-ism-sensors.md) | 10 | Sensor decoders: slicers (PWM, PPM, Manchester, NRZ), `sensor_decode` with readings as JSON, at least four 433 MHz and two 868 MHz devices (clean-room, no GPL code), flows `ism433_sensors`, `ism868_sensors` |
| M47 | [M47-morse.md](M47-morse.md) | 10 | Morse (CW) receive and transmit: `sdrk_proto::morse` (ITU-R M.1677-1 table, encoder, decoder with speed estimate), `morse_encode`, `morse_decode` (adaptive threshold), flows `cw_transmitter` and `cw_receiver` with tone variants |
| M48 | [M48-gauss-noise.md](M48-gauss-noise.md) | 10 | **Implemented** (2026-11-17; I48-1 closed by D3152, I48-2 by D3009). Kernel `gauss_noise` (`C32`/`CQ15`/`f32`/`Q15`): deterministic Gaussian noise from a seed with random access, one PCG32 draw per real value through a 4 KiB inverse-CDF table (bit-exact on every level, SIMD on every level), block and kind `noise`; first of the DSSS/OFDM kernel series (one kernel per milestone) |
| M49 | [M49-scramble.md](M49-scramble.md) | 10 | **Implemented** (2026-11-18, branch `m49-scramble`; no open items, I49-1 and I49-2 closed by the maintainer's decision D2092). Kernel `scramble` (`C32`/`CQ15`): keyed phase scrambling / descrambling by unit phasors from a caller-supplied index sequence; the quadrant part (`±1`, `±j`) exact by sign and swap (`b ≤ 2`, the DVB-S2 pattern, bit-exact), a residual `rotate` product only for `b ≥ 3`; introduces the shared exact sign operations (§A); SIMD on every level (revised 2026-11-17) |
| M49.1 | [M49.1-scramble-simplify.md](M49.1-scramble-simplify.md) | 10 | **Implemented** (2026-11-18, branch `m49-1-scramble-simplify`; the gap of `scramble_c32_b8` at `avx512`, 1.11 times the VOLK row on the `sapphirerapids` model, accepted by D2118). `scramble` for `bits ≥ 3`: one table phasor and one product per sample (the quadrant split stays for `bits ≤ 2`), a faster index production in the block `scramble`, fair reference rows (`bits = 8` against lookup + VOLK gated again; new `bits = 1` row) |
| M50 | [M50-xcorr-fft.md](M50-xcorr-fft.md) | 10 | **Implemented** (2026-11-23, D2600 to D2621). Kernel `xcorr_fft` (`C32`): overlap-save fast convolution / long matched filter for sync-word and spreading-code correlation (the backlog's "FFT fast convolution" item), bit-exact chunk invariance per backend, `from_spectrum` for a stored tap spectrum, SIMD composed per level; `CQ15` a concrete backlog item (revised 2026-11-17) |
| M51 | [M51-spread.md](M51-spread.md) | 10 | Kernel `spread` (`C32`/`CQ15`): direct-sequence spreading, short and long codes, chip phase tracked across calls; introduces the shared bit-packed code format `Code`/`CodeBits` and `Prbs::fill_words` (§B), binary chips by exact signs, complex chips by product; SIMD on every level (revised 2026-11-17) |
| M52 | [M52-despread.md](M52-despread.md) | 10 | Kernel `despread` (`C32`/`CQ15`): integrate-and-dump correlation (the `dot_complex` order) with running accumulators instead of a chip history, the matched receiver for `spread`; introduces the `CQ15` output shift (§C.1, default `ceil(log2 sf)`); SIMD on every level (revised 2026-11-17) |
| M53 | [M53-code-search.md](M53-code-search.md) | 10 | Kernel `code_search` (`C32`): parallel code-phase + Doppler acquisition (PCPS, after GNSS-SDR) with a double block (`K ≥ 2·L`), Doppler by bin shift (one forward FFT per dwell), GLRT and peak-to-floor statistics; SIMD composed per level as `channelize`; `CQ15` a backlog item (revised 2026-11-17) |
| M54 | [M54-code-track.md](M54-code-track.md) | 10 | Kernel `code_track` (`C32`): delay-locked loop (early/prompt/late, normalised E−L discriminator with unit slope), correlate-then-interpolate once per period with a fixed-point code position, prompt equals `despread` at zero bandwidth; SIMD for the lag sums and the per-period combination; `CQ15` a backlog item (revised 2026-11-17) |
| M55 | [M55-pilot-equalize.md](M55-pilot-equalize.md) | 10 | Kernel `pilot_equalize` (`C32`): OFDM pilot-based LS channel estimate, frequency interpolation, time smoothing, one-tap equalization by reciprocals, common-phase correction against the previous smoothed estimate, a defined zero threshold; SIMD on every level; `CQ15` a backlog item (revised 2026-11-17) |
| M56 | [M56-fwht.md](M56-fwht.md) | 10 | Kernel `fwht` (`C32`/`CQ15`): fast Walsh–Hadamard transform (Sylvester ordering), the spreading transform of MC-CDMA; `CQ15` per-stage rounding shift as the M07 FFT (scalings `Unscaled`/`Fixed`/`BlockFloat`, §C.2), reference in butterfly order; SIMD on every level; last of the DSSS/OFDM kernel series (revised 2026-11-17) |
| M57 | [M57-tetra-bsch.md](M57-tetra-bsch.md) | 10 | Planned (2026-11-28). TETRA downlink synchronization: the generic kernel `viterbi` (K 3 to 9, 2 to 4 outputs, `i16` modulo metrics, SIMD) with `conv_encode`, the bit helpers `Lfsr`, `MultiplicativeInterleaver`, `IndexPuncture` and `crc_bits` in `sdrk-proto`, `sdrk_proto::tetra`, the composite `tetra_bsch_decode` (frequency correction field search on phase steps, BSCH decoding), `tetra_parse` (SYNC PDU as JSON), flow `tetra_bsch_receiver` |
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
| M32 Block reference tests | M31 | M33–M36 |
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
| M44.1 QAM in building blocks | M44 (with the fix of I44-17), M45 | M46, M48–M56 |
| M44.1.1 One name per parameter | M44.1 | M46, M48–M56 |
| M44.2 QAM of any square order | M44.1.1 | M46, M48–M56 |
| M44.3 Equalizer switch and restart | M44.2 | M50–M56 |
| M44.4 SNR estimate, restart, dense blind start | M44.3 | M50–M56 |
| M13.1 Cycles only | M13 | everything (tooling only; planned specs updated) |
| M45 Channelizer, lane streams | M34 | M35–M44, M47 |
| M45.1 ISM band receiver | M45, M39.1 | M42, M43, M44 |
| M46 ISM sensor decoders | M45.1 | M42, M43, M44 |
| M47 Morse (CW) | M39.1 | M42, M43, M44, M45, M45.1, M46 |
| M48 `gauss_noise` | M38 | everything after M38 (the DSSS/OFDM kernels are independent of one another unless listed) |
| M49 `scramble` | M38 | everything after M38 |
| M49.1 `scramble` `bits ≥ 3` | M49 | everything after M49 |
| M50 `xcorr_fft` | M07, M43, M49 | everything after M43 and M49 |
| M51 `spread` | M38, M49 | everything after M49 |
| M52 `despread` | M43, M51 | everything after M51 |
| M53 `code_search` | M07, M51 | everything after M51 |
| M54 `code_track` | M38, M52 | everything after M52 |
| M55 `pilot_equalize` | M07, M45 | everything after M45 |
| M56 `fwht` | M29 | everything after M29 |
| M57 TETRA BSCH, `viterbi` | M13, M37, M38, M39.1, M40.1, M44.1.1 | everything after M44.1.1 |
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
| 620–625 | the golden gap cases of D356 (`resample_real` and `power_db` `Q15`, `convert_out` `rs8`/`ru8`; first numbered 469 to 474, which collided with M31 to M33, D357) |
| 626–635 | M45.1 (626–628 unused: the kernel cases moved to M45 with new seeds) |
| 636–649 | M46 |
| 650–659 | M47 |
| 660–679 | M45 |
| 680–689 | M48 (`gauss_noise`) |
| 690–699 | M49 (`scramble`) |
| 700–709 | M50 (`xcorr_fft`) |
| 710–719 | M51 (`spread`) |
| 720–729 | M52 (`despread`) |
| 730–739 | M53 (`code_search`) |
| 740–749 | M54 (`code_track`) |
| 750–759 | M55 (`pilot_equalize`) |
| 760–769 | M56 (`fwht`) |
| 770–774 | M46 extension: the sensor encoders (D1127) |
| 775–779 | M46 extension: the fact document (D1143; 775 `lacrosse_tx35`, 776 and 777 the corrected protocols, 778 the chunk plans of `pulse_key` in the FSK mode D1146, 779 the flow tests D1145) |
| 780–789 | M46 extension: the Ecowitt and Fine Offset devices (D1150 to D1163; 780 round trip, 781 flipped bits and truncated trains, 782 rejected readings, 783 250 kHz trains, 784 WS90 and WS85 without optional values, 785 their rejected readings, 786 the frame layout, 787 other frames, 788 battery steps, 789 JSON size) |
| 790–794 | M39.1 extension: ADS-B message parsing (D1430 to D1435; no seed is drawn, the tests start their generators from fixed constants; 790 to 794 are reserved for further cases) |
| 795–809 | free, for backlog milestones |
| 810–819 | M44 (I44-13 and I44-17) |
| 820–839 | M44.2 (QAM of any square order) |
| 840–859 | M44.1 (QAM in building blocks) |
| 860–869 | M44.1.1 (parameter names) |
| 870–879 | M49.1 (`scramble` for `bits ≥ 3`) |
| 880–889 | M44.2: the bit scramblers `bit_scramble` / `bit_descramble` (the maintainer's decision on I44.2-6; moved from 870–873 by the decision on I44.2-11, D1544: 880 and 881 the golden cases, 882 the cut plans, 883 the error propagation and the quarter turns; their round trip and self-synchronization use 838 and 839 of the range of M44.2) |
| 890–899 | M44.3 (the equalizer switch) |
| 900–909 | M44.4 (SNR estimate, restart, dense blind start) |
| 910–929 | M57 (TETRA BSCH, `viterbi`) |
| 930–999 | free, for backlog milestones |
| 1000+ | benchmark inputs (M13, M16) — not golden vectors |

## Files in this directory

- `M00-…` to `M57-…`: the milestones (each contains the full ground rules; from M26 on with the changed rule R1.1, from M37 on with `sdrk-proto` in R2). M48 to M56 are the DSSS/OFDM kernel series (one kernel per milestone with its primitive block and pipeline kind, for spread-spectrum and multicarrier work; the composite blocks and the flows follow in later milestones).
- `BACKLOG.md`: work after M18 and M44.
