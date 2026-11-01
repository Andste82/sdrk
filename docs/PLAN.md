# SDRk – Project Plan

> Snapshot of the project plan as of 2026-10-18. The living version is the shared plan document:
> https://claude.ai/code/artifact/694f271a-ba38-4b7c-b2fa-fb2e919d35c1
> If the two differ, the living document wins; update this snapshot when you notice a difference.
> Milestones derived from this plan: [`docs/milestones/README.md`](milestones/README.md).
>
> Status: phases 0 to 6 are delivered (milestones M00 to M15: kernels, blocks, pipeline, I/O, the SIMD backend, the WebAssembly bindings and the emulator targets). Phase 7 (milestones M16 to M18: ESP32-P4 PIE, RVV and Arm DSP backends) is planned, as are phases 8 and 9; the receive features of phase 9 are specified as milestones M19 to M25 (NFM with squelch, AM, SSB, audio filters, real-sampling devices, FM stereo, block names (M24.1), FM transmit audio); M19 to M24 and M24.1 are delivered. The release 0.1 (tag `v0.1.0`) is not made yet; it is milestone M25.1. Milestones M26 to M36 (block ports, composition, clock drift, primitive blocks, noise reduction, multi-input pipeline, AM/SSB and FM stereo transmit) extend phase 9, M37 to M44 are phase 10 (digital modes). From M26 on, milestones follow their declared dependencies and may run in parallel.

## Project

A Rust library of SDR building blocks for receiving and transmitting, from mixers and filters to complete demodulation chains, composed as JSON pipelines and controllable at runtime. The same kernels run heap-free on microcontrollers like the ESP32 and Cortex-M, on PCs and the Raspberry Pi, and as WebAssembly in the browser, each with dedicated SIMD optimizations from AVX-512 and NEON to the ESP32's PIE instructions. Sample-accurate timing, deterministic results on every platform, and a single test suite that runs in emulators and on real hardware make it a dependable foundation for firmware, desktop tools, web apps and AI-assisted signal analysis.

Name: **SDRk**, pronounced *shark* (Software-Defined Radio Kernels). SDRk is used in text, code comments and documentation; everything else in code is lowercase `sdrk`.

## Scope and principles

**Scope rule:** the repo contains only reusable SDR building blocks and what is needed to verify them. Applications, UIs, demos, services and device drivers live in their own repos.

1. **Kernels, not a SIMD abstraction:** a reference, a fallback and optional protokernels per backend.
2. **Four internal sample types:** `C32`, `CQ15`, `f32`, `Q15`; conversion only at the edges.
3. **Everything up to and including `blocks` is `no_std` and heap-free.** States are generic over their storage; heap only in `pipeline`, `io`, `bindings`.
4. **Traits per sample type;** the facade composes per kernel which backend runs it.
5. **Receive and transmit are equal citizens**, symmetric from day one.
6. **The sample index is the time base:** anchors for absolute time, rates as exact fractions, delays summed; the pipeline never reads a clock.
7. **Deterministic:** same backend, bit-identical output regardless of block size or partitioning; across backends the numeric contract's tolerance applies.
8. **Bindings configure, Rust processes:** flows as JSON (canonical), YAML (for humans) or via builders.
9. **Extensible without rework:** threads, GPU and network splits attach at node groups.
10. **Robust against untrusted input:** no panics, structured errors, resource limits.
11. **Adopt only foundations:** fearless_simd, num-complex, libm, bytemuck, heapless, serde, schemars. FFT and filter design are our own kernels.
12. **Drivers implement a stable interface** (`sdrk-device`) and live externally.
13. **License MIT/Apache-2.0;** no code taken from GPL projects.

## Structure

The workspace is grouped into layer folders; everything from `sdrk-blocks` down runs heap-free on microcontrollers.

```text
Outside this repo          applications · demos · device drivers · follow-up services
bindings                   wasm (c and python when needed)
sdrk-pipeline | sdrk-io    JSON/YAML, schema, build, runner, events | SigMF, WAV, raw
------------------------- heap boundary: below is no_std, no allocation -------------
sdrk-blocks                filter design, DDC/DUC, mod/demod, spectrum, descriptions
sdrk (facade)              picks the fastest available backend per kernel and platform
sdrk-backend-*             simd | arm-dsp | helium | rvv | esp-pie   (optional)
sdrk-core                  sample types, states, kernel traits, reference, fallback
sdrk-device                driver interface, SampleFormat, time types (stable, 1.0 early)
sdrk-testkit               beside every layer: generators, VirtualDevice, loopback, suite
```

Backends know only `sdrk-core`, never each other; only the facade knows all backends, and only the testkit knows the reference.

```text
sdrk/
├── crates/
│   ├── core/       sdrk-device, sdrk-core
│   ├── backends/   sdrk-backend-simd, -arm-dsp, -helium, -rvv, -esp-pie (as needed)
│   ├── facade/     sdrk
│   ├── blocks/     sdrk-blocks
│   ├── pipeline/   sdrk-pipeline            ← heap allowed from here up
│   ├── io/         sdrk-io
│   └── testing/    sdrk-testkit
├── bindings/       wasm/ (c/, python/ when needed)
├── schema/         generated JSON Schema (versioned)
├── targets/        test programs for emulators and boards
├── benches/        comparative benchmarks
├── tests/golden/   golden samples + generator script
├── examples/       fm-receiver, fm-transmitter (CLI)
├── xtask/          layer, heap, global-state, names and docs checks; bare-metal check; instruction counting (`count`); WASM package and tests; schema generation; emulator matrix (planned)
└── docs/           architecture, kernels, backends, device, pipeline, timing, performance
```

Folder = layer: `xtask` reads the layer from the path and fails the build on a forbidden dependency. Crate names keep the `sdrk-` prefix because they appear without their folder in manifests and on crates.io.

## Phases 0–3: from foundation to a working pipeline

Phases 0 to 3 are the core: after them the library is usable, tested and produces audible results in both directions.

### Phase 0 – Foundation (small)

- Confirm name (crates.io, GitHub, domain), license, minimum Rust version; scope rule in the README
- Workspace with the agreed folder structure: `sdrk-device`, `sdrk-core`, `sdrk`, `sdrk-testkit`
- `xtask`: layer check (folder = layer), heap boundary, ban on global mutable state, bare-metal check (`thumbv7em-none-eabihf`)
- `architecture.md` with principles and extensibility rules; `performance.md` with performance targets
- CI for x86 with cargo deny (security advisories and dependency licenses: no GPL in the library); rustdoc with examples required for public items; semver policy, changelog and crate publishing order in CONTRIBUTING.md

**Done when:** the workspace builds on x86 and as `no_std`, and all checks run in CI.

### Phase 1 – Core foundations (medium)

- **`sdrk-device`:** `SampleFormat`; RX and TX streams; per-block metadata (sample index, gap, overflow/underrun, burst start/end, start time); time types (u64 sample index, rate as fraction, time anchor as integer ns + source)
- **`sdrk-core`:** sample types, states with `reset()`, kernel traits per category and sample type, history buffer for sliding windows, helper macro for per-sample kernels
- **Facade:** per-kernel composition, initially fallback only
- **Q15 conventions** (rounding, saturation, scaling), aligned with CMSIS-DSP
- **Kernels:** f64 reference and fallback for `convert`, `rotate`, `fir`
- **Rules:** everything `Send` (compile-time check); no panics
- **Testkit:** PRNG, `tone`, `noise`, `VirtualDevice` (RX/TX, anchors, retune events), `Compare`, block-size invariance, `suite()`
- **Tests:** math tests, exhaustive `convert`, proptest, extreme inputs (silence, full scale, NaN)
- **Docs:** determinism promise, first part of `timing.md`

**Done when:** the suite passes on the PC.

### Phase 2 – MVP kernels, both directions (large)

- **Receive:** `convert` extended to all input formats incl. packed 10/12-bit and MSB/LSB alignment, `fir_decimate`, `resample`, `quad_demod`, `deemphasis`, `window`, `mag_sq`, `power_db`
- **Front-end correction:** `dc_block` (needed for almost every RTL-SDR recording)
- **Transmit:** `convert_out` (rounding, saturation, packing), `fir_interpolate`, `fm_mod`
- **Own FFT:** reference (naive DFT), radix-2 fallback in f32 and Q15 (fixed and block-floating-point scaling), complex/real/inverse, static twiddles; spectrum blocks account for the block exponent (+6.02 dB per shift)
- **Fused kernels** with default implementation: `ddc`, `duc`, `fm_demod`. A fused variant is kept only if its instruction count beats the chained single kernels (lesson from GNU Radio 4); otherwise the default chaining stays.
- Every kernel declares its rate ratio (fraction, constant/variable) and group delay
- **Golden samples** from SciPy and GNU Radio; numeric contracts in `kernels.md`
- **Testkit:** FM generator, `SinadMeter`, `SpectralMask`, `ToneMeter`; weak-tone test (−60 dBFS) for block-floating-point FFT

**Done when:** a synthetic FM broadcast reaches the SINAD threshold in f32 and Q15, and references match the golden samples.

### Phase 3 – Blocks, pipeline, files (large)

- **`sdrk-blocks`:** filter design (checked against SciPy), DDC, DUC, FM demodulator with audio low-pass, FM modulator, resampler (`cutoff_hz`, ppm fine-tuning), spectrum; each block with a **mandatory description** (purpose, ports, parameters with unit, range, typical values, runtime/build-time)
- **`sdrk-pipeline`:**
  - Data model with serde; JSON canonical, YAML for authoring (feature, not in the WASM build); `description` fields
  - Registry → versioned JSON Schema (schemars)
  - Build: type checks, rate propagation, automatic input conversion, delay/lead-time, buffer sizing by latency and cache
  - Validate-only call; resource limits and cost estimate; machine-readable errors with node path
  - Execution: push mode (RX) and pull mode (TX), zero-copy fan-out, ring buffer per edge, reset on gaps
  - Inputs: borrowed data without copy; joining mid-stream (start sample index, inherited anchors)
  - Events: gap, overflow/underrun, time anchor, retuned, burst, start time, detection, parameter acknowledgement
  - Parameters: immediate or at a sample index; validation callback per block; acknowledgement with effective sample index
  - Outputs: stream or frames with metadata; drop policy; latency per output; intervals in samples
  - Introspection: current config as lossless JSON; runtime statistics per node
  - Backend preference/requirement per node; node groups in the data model (no effect yet); edges and group transitions as interfaces; wire format for split pipelines specified (versioned)
- **`sdrk-io`:** SigMF incl. annotations, WAV, raw formats
- **`sdrk-device`:** control capabilities (frequency, rate, gain, bandwidth), hardware timestamps, timed commands, transmit, error kinds; async variant compatible with WebUSB/Web Serial; checked against SoapySDR
- **Testkit:** loopback with channel simulator (delay incl. fractional, noise, frequency offset, attenuation)
- **Examples:** `fm-receiver` (JSON flow + recording → WAV), `fm-transmitter` (WAV → cs8 file)
- **Tests:** block-size invariance for whole pipelines; pipeline = hand-wired chain; JSON round-trip; schema validation of all example flows; FM over loopback (SINAD); burst timing over loopback (±1 sample); two stations without crosstalk; exact event propagation; long runs without drift; fuzzing of parser and build
- **Architecture probe:** a two-thread runner as a test; multi-threaded = single-threaded, bit-identical
- **Paper check:** GPU group and network split describable in `architecture.md`
- **Benchmark:** latency in samples (from buffer sizes and group delays) and throughput as instructions per sample, reported together

**Done when:** synthetic recordings are demodulated via a JSON flow, and all loopback tests pass. No real recordings are needed: correctness rests on golden samples from established implementations.

## Phases 4–9: speed, platforms, bindings

### Phase 4 – SIMD backend and comparative benchmarks (large)

- `sdrk-backend-simd`: `convert`, `rotate`, `fir`/`fir_decimate`/`fir_interpolate`, `mag_sq`, `quad_demod`, `fm_mod`
- SIMD FFT (radix-4, vectorized butterflies, modelled on rustfft)
- Comparative benchmarks against VOLK and rustfft (dev-dependencies only)
- Instruction counting in CI (`cargo xtask count`, Valgrind callgrind, Intel SDE, QEMU and wasmtime)
- CI matrix: x86 native, AVX-512 via Intel SDE, AArch64 via QEMU, WASM under wasmtime (with/without SIMD128, with WASM's default stack limit to catch oversized stack buffers)

**Done when:** the suite passes everywhere and the instruction-count target (at most 1.10× the reference, decision D14; [performance.md](performance.md)) is met. Then release 0.1.

### Phase 5 – WASM bindings (small to medium)

- Pipeline API: JSON, builder, parameters, push and pull, outputs, introspection
- Two builds (with/without SIMD128), selected in JavaScript; relaxed SIMD only as an opt-in feature, without the bit-exactness promise
- Docs: Web Workers, cross-origin isolation for `SharedArrayBuffer`
- Tests: `wasm-bindgen-test` in Node and headless Chrome and Firefox
- npm package; demos live in other repos

**Done when:** tests pass in Node and both browsers, and the package installs.

### Phase 6 – Embedded in emulators (medium)

- `targets/qemu-cortex-m4`, `targets/esp32p4` (esp-emulator), `targets/esp32s3` (Espressif QEMU fork) running `testkit::suite`, both directions
- Test profiles: quick (every commit) and full (nightly/release); tier-1 platforms on every commit, tier-2 nightly
- `xtask test-all` locally and in CI

**Done when:** the suite passes in all three emulators.

### Phase 7 – Embedded backends (large, incremental)

- All embedded backends are developed and verified in emulators; real boards are optional and come later (phase 9)
- First platform: ESP32-P4 PIE in esp-emulator; the most expensive kernels, including Q15 FFT
- Instruction-count comparison against ESP-DSP and CMSIS-DSP, with the method per platform from *Testing and performance*
- Compile-time composition via generics, heap-free, for MCU pipelines
- Second platform: RISC-V Vector (`rvv`, vector-length agnostic, tested at VLEN 128–1024 in QEMU; bit-identical results for every vector length; milestone M17)
- Third platform: Arm DSP extension for Cortex-M4/M7/M33 (`arm-dsp`, bit-exact Q15 with 64-bit accumulators, compared with CMSIS-DSP; milestone M18)
- Then only on concrete demand: `helium` (M55/M85, small installed base) and S3 PIE (only after the emulator is shown to execute the `ee.*` instructions; see BACKLOG)

**Done when:** per backend, the suite passes in the emulator and the instruction-count target is met.

### Phase 8 – Further bindings (optional, when needed)

- C bindings (pipeline API and builder, generated header) and Python bindings (PyO3, NumPy without copy), added only when a concrete use needs them
- Cheap to retrofit because the public API stays binding-friendly from phase 3 on: plain types, errors instead of panics, fixed sample-type variants next to generics; the WASM bindings keep these rules tested

### Phase 9 – Stabilization and extensions

- Driver conformance suite complete; `sdrk-device` 1.0
- **More kernels and blocks:** `iq_balance`, `real_to_complex` (real-sampling devices such as Airspy), `sigma_delta_decimate` (SX1255 mode A), `biquad`, CIC, halfband, AGC, squelch, AM/SSB, pulse shaping (RRC), FM stereo (M24) and RDS
- **Analysis blocks** with structured results: signal detection (frequency, bandwidth, level, SNR), occupied bandwidth, noise floor, burst detection, instantaneous values and histograms, symbol-rate estimator
- **Digital building blocks:** generic OOK/ASK/FSK/PSK demodulators with clock recovery, bit slicer, preamble correlator, diagram data as frames (constellation, eye diagram)
- **Real hardware:** the same `targets/` on boards via probe-rs, cycle counters, real-time targets per platform
- **Optional:** GStreamer-style shorthand for flows; compiling a JSON flow into heap-free Rust code
- **Composition and analog transmit (M26–M36):** typed block ports, `Chain`/`Fork` composition with heap-free storage, primitive blocks and plans that equal the specific blocks bit for bit, clock-drift control, noise blanker and automatic notch, two-input nodes in the pipeline, AM/SSB and FM stereo transmit

### Phase 10 – Digital modes

- Packet ports and the crate `sdrk-proto` (protocol logic without signal processing), symbol timing recovery, slicer, test channel and BER meter (M37, M38)
- FSK/GFSK with AFC, AFSK/APRS, OOK, ADS-B (M39, M39.1); BPSK/QPSK and variants (M40, M40.1); RDS receive and transmit (M41, M42); adaptive equalizer and QAM (M43, M44)
- Receive and transmit of a mode in the same milestone, with loop tests TX → channel → RX against the theoretical bit error rate

### Later (prepared, not scheduled)

- Multi-threaded runner (sleeps when idle), node groups on MCU cores
- Data-parallel blocks for OFDM and channelizers
- Network splits using the specified wire format
- GPU groups (e.g. wgpu, VkFFT), first as an experiment in a separate repo

## Core concepts

### Pipeline

- **Model:** nodes (block instances), ports (type, rate), edges, pipeline inputs/outputs, parameters, groups.
- **Lifecycle:** describe → validate → build → run ⇄ control. No memory is allocated after the build.
- **Formats:** JSON is canonical and validated by a generated, versioned JSON Schema; YAML for hand-written flows; builders in host languages produce the same model.
- **Validation in two stages:** the schema checks structure, types and ranges; the build checks edge types, rates, cycles and dangling outputs, with errors per node.
- **Execution:** synchronous, single-threaded, fixed order; small ring buffers per edge sized for latency and cache; zero-copy fan-out.
- **Not supported by design:** threads inside the core runner, feedback in the graph (loops live inside blocks), topology changes while running (rebuild instead).
- **Host languages configure, Rust processes:** data enters once and results leave once per block of samples.

### Time

- The **sample index** (u64) is the time base; absolute time comes from **anchors** (integer ns + source: device, host, GPS).
- **Rates are exact fractions;** every block declares its ratio (constant or variable) and group delay; the pipeline sums delays on receive and computes lead time on transmit.
- **Events** sit at sample positions (gap, overflow/underrun, anchor, retuned, burst, start time, detection, parameter applied) and are moved through the chain by the pipeline, not by kernels.
- **Intervals** (e.g. a spectrum every 33 ms) are converted to sample counts at build time; the pipeline never reads a clock.
- **Parameter changes** can take effect at a given sample index, independent of block boundaries.

### Transmit

- **Pull mode:** the sink sets the pace; the pipeline produces exactly N samples on request.
- **Continuous and burst transmission;** burst start/end, optional start time, underrun and "too late" reported as events.
- **Symmetric building blocks:** `convert_out`, interpolation, DUC, modulators.
- **Loopback** in the testkit (channel simulator) tests TX and RX against each other without hardware.

### Extensibility

- **One extension point, the node group,** serves threads, MCU cores, GPU and network splits; kernels, backends and blocks stay untouched.
- **Guarded by rules:** no global mutable state, everything `Send`, edges as interfaces, determinism independent of partitioning.
- **Verified early:** a two-thread architecture probe in phase 3 must produce bit-identical results.

## Testing and performance

Correctness rests on a chain of trust; performance is measured against established libraries at equal accuracy.

**Chain of trust:** math properties and golden samples (SciPy, GNU Radio, CMSIS-DSP conventions) validate the f64 reference; the reference validates the fallback and every backend; end-to-end measurements and loopback check plausibility.

| Where | What runs |
| --- | --- |
| PC | golden samples, math tests, proptest, fuzzing, suite, pipeline and loopback tests, schema validation, architecture probe |
| WASM | suite under wasmtime; bindings in Node and headless browsers |
| Emulators | suite with fixed seeds, small end-to-end test (QEMU, Intel SDE, esp-emulator, Espressif QEMU) |
| Real boards | the same suite via probe-rs |
| External drivers | conformance suite from the testkit |

Q15 results are bit-exact for simple kernels and bound to a minimum SNR for filters, FFT and demodulation.

**Performance is measured by counting executed instructions in the same emulator**, for our kernel and for the established library, with identical input size and build flags. Instruction counts are deterministic, so there is no measurement noise. Speed in the core is added by static simulation, not by real-hardware timing (see the measurement policy below).

| Platform | Counting tool | Compared against |
| --- | --- | --- |
| x86-64 | callgrind (read by `cargo xtask count`); Intel SDE instruction mix for SSE2, SSE4.2 and AVX-512 | VOLK, rustfft |
| AArch64 | QEMU user mode with the instruction-count plugin | VOLK |
| Cortex-M4/M33/M55 | QEMU system mode with the instruction-count plugin | CMSIS-DSP |
| ESP32-P4/S3 | S3: QEMU instruction-count plugin in the Espressif QEMU fork (plugin-enabled build; plugin support and PIE emulation unverified, otherwise static counting as for the P4); P4: static instruction count of the inner loops from the disassembly, as esp-emulator only samples | ESP-DSP |
| RISC-V V | QEMU with the instruction-count plugin | own scalar fallback (no established library) |
| WASM | wasmtime fuel metering | rustfft (`wasm_simd`) for the `C32` FFT and `rfft`, KissFFT for the `CQ15` FFT, VOLK compiled for wasm32 for `mag_sq` and the conversions (its other kernels as information, decision D12); our fallback for the kernels without an established library |

| Instructions per sample vs. reference library | Meaning |
| --- | --- |
| ≤ 1.00× | goal: equal or better (marked in the results) |
| ≤ 1.10× | release limit; CI fails above it for a `release` entry (decision D14) |
| ≤ 1.50× | allowed only while a kernel is under development (`dev`), never in a release; `--strict` (the gate of the 0.1 tag) fails for every entry that is not a clean `release` entry |

`docs/performance.md` and `benches/limits.json` are the current contract; they supersede these tables where they differ.

Comparisons hold accuracy at least equal (within 1 dB or better), shown by the error against the f64 reference. Instruction counts are a strong proxy on in-order cores (Cortex-M, ESP32) and a weaker one on out-of-order CPUs and for memory-bound kernels.

**Measurement policy (D32 to D34, issue #20; details in [performance.md](performance.md#measurement-policy) and [kernels.md](kernels.md#accuracy-policy)):**

- **Fair comparison.** Every SIMD branch is compared with an established library at the same level, with the same algorithm class and the same work; the fallback with established scalar code. SIMD against our own fallback is at most `info`, never a gate. Without any fair reference, the efficiency of the branch is judged instead.
- **Speed, not only work.** Instructions measure work. Once level 2 is built, the branch of each instruction-set level on out-of-order cores is judged by cycles per sample simulated with `llvm-mca` (one reference core model and one counter-model per level, ratios only, no real hardware, no calibration), together with its efficiency (port bound / cycles). Instructions stay the gate on in-order cores and WASM, and everywhere until level 2 exists.
- **Accuracy against the optimum.** The f64 golden is the mathematical optimum and the first target, for float and Q15 kernels. An established library is a yardstick only where the optimum cannot be reached by principle or data type. Regression guards close to the measured SNR complement the loose contract floors.

## Outside this repo and follow-up projects

Two follow-up projects build on the library, each in its own repo with its own plan; the library only guarantees the foundations they need.

**Outside this repo:**

- **Device drivers** (RTL-SDR, HackRF, SoapySDR, LR2021, SX1255, ESP-SDR …), implementing `sdrk-device`. Recommendation: driver logic separate from transport — native via `nusb`, browser via WebUSB/Web Serial.
- Browser demos, desktop apps, graphical flow editors
- FutureSDR integration

### Follow-up 1 – SDR agent service

A long-running service with REST/WebSocket and an MCP adapter, through which an AI agent creates, changes and deletes pipelines, sets parameters, and fetches measurements and detections.

- The agent never sees samples, only measurements, detections and rendered diagrams.
- Many small pipelines; "change" means rebuilding one of them.
- One device stream is fanned out to many pipelines.
- Recordings and live signals are treated the same.
- Transmit is off by default.
- **Starts after:** phase 3. **Prior art:** AetherLink, signallm, sdr-mcp, SDRangel REST API.

### Follow-up 2 – Interactive signal analysis

A server with an SDR, a web UI and MCP access: mark a signal in the waterfall, the agent investigates, demodulates and decodes it, and every step is visible live. Builds on follow-up 1.

- One shared state on the server; UI and agent use the same API and receive changes over WebSocket.
- Activity log with author (human or agent); human takes precedence; emergency stop and undo.
- Markers (frequency and time range) as numbered objects.
- IQ ring buffer (30–60 s); markers saved as SigMF excerpts with annotations.
- Iterate on the recording first, then switch the working chain to live.
- Diagrams rendered for the agent (waterfall excerpt, constellation, eye diagram, histogram) plus measurements as JSON.
- Spectra and decimated channels over the network instead of raw IQ; the split between server and browser is configurable.
- Transmit off by default; legal limits documented (no interception of traffic not addressed to you).
- **Starts after:** phase 3; analysis features after phase 9.

### Requirements on the library

| Requirement | Phase |
| --- | --- |
| Pipelines from JSON/YAML with versioned schema | 3 |
| Registry with mandatory descriptions | 3 |
| Validate-only call, machine-readable errors | 3 |
| Resource limits, cost estimate | 3 |
| No panics, fuzzing | 1, 3 |
| Runtime parameters with acknowledged effective sample | 3 |
| Joining mid-stream, borrowed inputs | 3 |
| Lossless config readout, runtime statistics | 3 |
| Events with sample index and time anchor | 3 |
| Frames with metadata | 3 |
| Wire format for split pipelines (specification) | 3 |
| SigMF annotations | 3 |
| Independent pipelines, everything `Send` | 1, 3 |
| Determinism (recording = live) | 1–3 |
| Analysis and digital building blocks | 9 |

## Decisions

| Topic | Decision |
| --- | --- |
| Name | **SDRk** in text, code comments and documentation, pronounced shark (Software-Defined Radio Kernels); lowercase `sdrk` everywhere else in code |
| Support tiers | Tier 1 (every commit, a failure blocks the merge): x86-64, AArch64, wasm32, Cortex-M4. Tier 2 (nightly): RISC-V, Cortex-M55; ESP32-P4/S3 paused in CI (D201). Tier 3 (builds, not tested): every other Rust target via the fallback |
| Q15 format | Q1.15 in `i16`, range −1 to 1 − 2⁻¹⁵ |
| Q15 rounding | round half up: add ½ LSB, then arithmetic shift |
| Q15 overflow | always saturate, never wrap (−1 × −1 saturates to the maximum) |
| Q15 multiply | (a · b + 2¹⁴) >> 15, saturated |
| Q15 accumulation | reference and fallback in `i64`; fast variants may use `i32` with documented headroom; round and saturate once at the end |
| Q15 FFT scaling | two modes from the start: fixed (halve after every stage in both directions, output scaled by 1/N, cannot overflow for inputs with |x| ≤ 1; matches CMSIS-DSP; forward then inverse returns x/N) and block floating point (halve only when overflow threatens, one shared exponent per block, returned with the result; use it when precision matters, e.g. round trips); one exponent rule for every type and direction: unnormalized sum = output · 2^exponent (C32 forward 0; C32 inverse and Q15 fixed log2 N; BFP = sum of the per-stage shifts); FFT only, not streaming filters |
| f32 → Q15 | round and saturate; NaN → 0, ±∞ → ±maximum |
| f32 denormals | feedback states (IIR, deemphasis, AGC, PLL) are flushed to zero below 1e-30; no reliance on CPU flush-to-zero modes, so results stay identical across backends |
| Byte order | little-endian targets only (all current targets are); zero-copy reinterpretation relies on it; big-endian is not supported |
| Bit-exactness | reference and fallback always; accelerated variants bit-exact for simple kernels, minimum SNR otherwise; WASM relaxed SIMD is opt-in and only bound to the numeric contract's tolerance |
| Fused kernels | kept only if their instruction count beats the chained single kernels |
| Performance target | instruction count in the emulator vs. the established library: goal ≤ 1.00×, release limit ≤ 1.10×, dev ≤ 1.50× (decision D14); on x86-64 and AArch64 simulated cycles per sample once level 2 exists (D33) |
| Counting on ESP32 | S3: QEMU instruction-count plugin (plugin-enabled build of the Espressif fork); P4: static count of the inner loops from the disassembly; exact counting in esp-emulator would be a welcome upstream contribution |
| Real-time targets | deferred until real hardware is in use; the speed judgement of level 2 (D33) uses simulation only |
| YAML crate | `serde-saphyr` (pure Rust, no `unsafe-libyaml`), behind the `yaml` feature, not in the WASM build; version pinned at the start of phase 3 |
| Dependency hygiene | `cargo deny` in CI for security advisories and licenses; no GPL dependencies in the library |
| Releases | semver; `sdrk-device` reaches 1.0 early, other crates stay 0.x until phase 9; changelog and publishing order in CONTRIBUTING.md |
