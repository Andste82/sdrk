# Backlog — work after M18 and M23

> Plan: [PLAN.md › Phases 7–9 and Later](../PLAN.md#phases-49-speed-platforms-bindings) · Milestone index: [README.md](README.md)

These items are planned but not yet written as implementation-ready milestones. When one is scheduled, turn it into a milestone file `M19-…md` (and so on) using the structure of the existing milestones (goal, why, ground rules marker, scope, specification, test vectors with seeds, tests, acceptance, deliverables) and the ground rules block. Keep the order roughly as listed; dependencies are noted.

## Phase 7 — further embedded backends (same recipe as M13/M16)

The RISC-V Vector backend is milestone [M17](M17-rvv-backend.md), the Arm DSP-extension backend (Cortex-M4/M7/M33, also the RP2350) is [M18](M18-arm-dsp-backend.md). Left for the backlog from it: 32-bit RISC-V with `V` and the embedded subsets `Zve32f`/`Zve64f` (no 64-bit lanes: the CQ15 filters need a chunked or fallback path), vectorised `quad_demod` and `fm_mod`.


| Item | Content | Counting / comparison | Depends on |
| --- | --- | --- | --- |
| `sdrk-backend-helium` | Cortex-M55/M85 MVE: Q15 and f32 kernels. **Low priority, only on concrete demand:** the chips (Alif Ensemble, STM32N6, Renesas RA8) are far less widespread than Cortex-M4/M7. Emulation is available: QEMU supports Armv8.1-M Helium and the Cortex-M55 since 6.0 ([release notes](https://linuxiac.com/qemu-6-0-0-released-with-focus-on-arm-and-risc-v/)), and the machine `mps3-an547` is used in practice (e.g. Zephyr CMSIS-NN tests). Counting as in M18 (one vector instruction counts as one; MVE executes in overlapping beats on real cores, so counts show instruction economy only). Intrinsics are not stable in Rust: whole-function assembly as in M17/M18 | QEMU `mps3-an547` + plugin vs CMSIS-DSP Helium variants | M18 |
| ESP32-S3 PIE | `ee.*` instructions (Xtensa LX7, eight 128-bit `QR` registers for 8/16/32-bit elements, `QACC` accumulators of 2 × 160 bit, 40-bit `ACCX`; different from the P4's RISC-V PIE); kernels as M16. **Precondition (unverified in this plan): the Espressif QEMU fork must emulate the `ee.*` instructions.** Its documentation lists emulated peripherals only and says nothing about PIE ([ESP-IDF QEMU page](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/tools/qemu.html), [fork README](https://github.com/espressif/esp-toolchain-docs/blob/main/qemu/README.md)); Espressif's [PIE introduction](https://developer.espressif.com/blog/2024/12/pie-introduction/) names no emulator or assembler, and the forum reports PIE instructions missing from the TRM. Whether TCG plug-ins work for Xtensa in the fork is likewise unconfirmed. **Start with a smoke test** (execute a few `ee.*` instructions in the fork and compare the results with the TRM); if they are not emulated, the S3 waits for hardware (phase 9) or for emulator support, because the plan's rule is emulator first. Counting: plug-in if it works, otherwise static from the disassembly as in M16. Needs the Espressif Rust toolchain (`espup`) | Espressif QEMU fork vs ESP-DSP `_aes3` variants | M16 |
| Compile-time MCU pipelines | generic, heap-free composition of blocks into a fixed chain (`Chain<A, B>` with typed edges and caller storage), tested in `targets/` | — | M15 |

## Phase 8 — further bindings (optional, only on concrete demand)

- **C bindings** (`bindings/c`): `extern "C"` API mirroring the WASM API (pipeline from JSON, push/pull, outputs, events, params), opaque handles, error codes = `ErrorKind` numbers, header generated with `cbindgen` (checked in, verified by CI).
- **Python bindings** (`bindings/python`): PyO3 + maturin, NumPy arrays without copy (`numpy` crate), same API shape; wheels for the three host platforms.
- Rule since M10: the public API stays binding-friendly (plain types, errors instead of panics, fixed sample-type variants next to generics); the WASM bindings keep this tested.

## Phase 9 — stabilization and extensions

- **Platform-independent golden vectors:** the generator produces other bits on Windows and Linux arm64 in about 40 cases (Windows: `fft` 18, `design` 6, `rotate` 5, `fm_mod` 4, `window` 3, `blocks` 2, `quad_demod` 2, `power_db` 1; arm64: `fft` 18, `design` 3, `blocks` 2, `fir` 2, `fir_decimate` 2, `window` 1; CI run 40194336676), because NumPy and SciPy use the math library of the platform. Compute the transcendental functions, the FFT reference and the filter design of those cases with own scalar code (`sdrk_golden/scalar_math.py`, as M09 did for `dot`, `blocks` and `kaiserord`), then make `golden_x64_win` and `golden_arm64_linux` compare byte for byte and drop the Linux-only rule of `tests/golden/README.md`. Maintainer decision of 2026-10-25: do this next.

- **Driver conformance suite complete** (all capabilities of M12, timed commands, async streams) and **`sdrk-device` 1.0** (API review, semver promise, MSRV policy).
- **Kernels and blocks** (squelch, AGC, AM, SSB, `biquad`, halfband decimation and `real_to_complex` are the milestones [M19](M19-nfm-squelch.md) to [M23](M23-real-sampling-input.md)): `iq_balance` (IQ imbalance correction), `sigma_delta_decimate` (SX1255 mode A, 1-bit `Bits1` container from M01), CIC (decimate/interpolate), halfband interpolation by 2, pulse shaping (RRC), FM preemphasis (transmit), FM stereo and RDS, multi-input nodes (adder, mixer, multiplexer) in the pipeline.
- **FFT extensions:** Q15 real FFT and its inverse, FFT-based fast convolution (overlap-save) as a block.
- **Analysis blocks** with structured results (frames/events): signal detection (frequency, bandwidth, level, SNR), occupied bandwidth, noise floor, burst detection, instantaneous values and histograms, symbol-rate estimator.
- **Digital building blocks:** generic OOK/ASK/FSK/PSK demodulators with clock recovery, bit slicer, preamble correlator, diagram data as frames (constellation, eye diagram).
- **Real hardware:** the same `targets/` on boards via `probe-rs` (Cortex-M, ESP32 via `espflash`), DWT cycle counters / `mcycle`, real-time targets per platform (the plan defers them until hardware is in use).
- **Device API gaps** identified by the SoapySDR mapping in M12 (sensors, generic settings, frequency components, clock/time source selection).
- **Optional:** GStreamer-style shorthand for flows (`rf ! ddc frequency_hz=600e3 ! fm_demod ! out`) compiled to the JSON model; compiling a JSON flow into heap-free Rust code for MCUs.
- **Release work:** check that `sdrk` is free on crates.io and npm, crates.io publishing in the order given in CONTRIBUTING.md, npm package name.

## Later (prepared, not scheduled)

- Multi-threaded runner that sleeps when idle, using node groups (M11) as the partitioning unit; node groups on MCU cores.
- Data-parallel blocks for OFDM and channelizers (polyphase filter bank channelizer).
- Network splits using the wire format (M11) over TCP/WebSocket.
- GPU groups (e.g. wgpu, VkFFT), first as an experiment in a separate repository.

## Golden-vector seed ranges

New cases must use seeds from a free range (see the table in [README.md](README.md#seed-ranges-for-golden-vectors)); the next free range for backlog items starts at **530**.
