# Backlog — work after M16

> Plan: [PLAN.md › Phases 7–9 and Later](../PLAN.md#phases-49-speed-platforms-bindings) · Milestone index: [README.md](README.md)

These items are planned but not yet written as implementation-ready milestones. When one is scheduled, turn it into a milestone file `M17-…md` (and so on) using the structure of the existing milestones (goal, why, ground rules marker, scope, specification, test vectors with seeds, tests, acceptance, deliverables) and the ground rules block. Keep the order roughly as listed; dependencies are noted.

## Phase 7 — further embedded backends (same recipe as M13/M16)

| Item | Content | Counting / comparison | Depends on |
| --- | --- | --- | --- |
| `sdrk-backend-arm-dsp` | Cortex-M4/M7/M33 DSP extension (SIMD32: `SMLAD`, `QADD16`, …) for CQ15 `fir`, `fir_decimate`, `dot`, `mag_sq`, `convert`, `fft` | QEMU system mode + instruction-count plugin vs **CMSIS-DSP** (`arm_fir_q15`, `arm_fir_decimate_q15`, `arm_cfft_q15`, `arm_cmplx_mag_squared_q15`), built in a separate `benches/` project | M15 |
| `sdrk-backend-helium` | Cortex-M55/M85 MVE: Q15 and f32 kernels | QEMU `mps3-an547` + plugin vs CMSIS-DSP Helium variants | arm-dsp |
| `sdrk-backend-rvv` | RISC-V Vector 1.0, vector-length agnostic; test with VLEN 128, 256, 512 (`qemu-riscv64 -cpu rv64,v=true,vlen=…`) | QEMU plugin vs own scalar fallback (no established library) | M15 |
| ESP32-S3 PIE | `ee.*` instructions (different from P4); kernels as M16 | Espressif QEMU fork, plugin-enabled build, vs ESP-DSP `_aes3` variants | M16 |
| Compile-time MCU pipelines | generic, heap-free composition of blocks into a fixed chain (`Chain<A, B>` with typed edges and caller storage), tested in `targets/` | — | M15 |

## Phase 8 — further bindings (optional, only on concrete demand)

- **C bindings** (`bindings/c`): `extern "C"` API mirroring the WASM API (pipeline from JSON, push/pull, outputs, events, params), opaque handles, error codes = `ErrorKind` numbers, header generated with `cbindgen` (checked in, verified by CI).
- **Python bindings** (`bindings/python`): PyO3 + maturin, NumPy arrays without copy (`numpy` crate), same API shape; wheels for the three host platforms.
- Rule since M10: the public API stays binding-friendly (plain types, errors instead of panics, fixed sample-type variants next to generics); the WASM bindings keep this tested.

## Phase 9 — stabilization and extensions

- **Driver conformance suite complete** (all capabilities of M12, timed commands, async streams) and **`sdrk-device` 1.0** (API review, semver promise, MSRV policy).
- **Kernels and blocks:** `iq_balance` (IQ imbalance correction), `real_to_complex` (real-sampling devices such as Airspy: fs/4 shift + halfband), `sigma_delta_decimate` (SX1255 mode A, 1-bit `Bits1` container from M01), `biquad`, CIC (decimate/interpolate), halfband (decimate/interpolate by 2), AGC, squelch, AM and SSB demodulators, pulse shaping (RRC), FM preemphasis (transmit), FM stereo and RDS, multi-input nodes (adder, mixer, multiplexer) in the pipeline.
- **FFT extensions:** Q15 real FFT, inverse real FFT (f32 and Q15), FFT-based fast convolution (overlap-save) as a block.
- **Analysis blocks** with structured results (frames/events): signal detection (frequency, bandwidth, level, SNR), occupied bandwidth, noise floor, burst detection, instantaneous values and histograms, symbol-rate estimator.
- **Digital building blocks:** generic OOK/ASK/FSK/PSK demodulators with clock recovery, bit slicer, preamble correlator, diagram data as frames (constellation, eye diagram).
- **Real hardware:** the same `targets/` on boards via `probe-rs` (Cortex-M, ESP32 via `espflash`), DWT cycle counters / `mcycle`, real-time targets per platform (the plan defers them until hardware is in use).
- **Device API gaps** identified by the SoapySDR mapping in M12 (sensors, generic settings, frequency components, clock/time source selection).
- **Optional:** GStreamer-style shorthand for flows (`rf ! ddc frequency_hz=600e3 ! fm_demod ! out`) compiled to the JSON model; compiling a JSON flow into heap-free Rust code for MCUs.
- **Release work:** check that `sdrk` is free on crates.io and npm, crates.io publishing in the order given in CONTRIBUTING.md, npm package name.
- **Test data:** an optional small real recording (≤ 5 MiB, license CC0/CC-BY, SigMF) for the `real_recording` system test.

## Later (prepared, not scheduled)

- Multi-threaded runner that sleeps when idle, using node groups (M11) as the partitioning unit; node groups on MCU cores.
- Data-parallel blocks for OFDM and channelizers (polyphase filter bank channelizer).
- Network splits using the wire format (M11) over TCP/WebSocket.
- GPU groups (e.g. wgpu, VkFFT), first as an experiment in a separate repository.

## Golden-vector seed ranges

New cases must use seeds from a free range (see the table in [README.md](README.md#seed-ranges-for-golden-vectors)); the next free range for backlog items starts at **410**.
