# SDRk

A Rust library of SDR building blocks for receiving and transmitting, from mixers and filters to complete demodulation chains, composed as JSON pipelines and controllable at runtime. The same kernels run heap-free on microcontrollers like the ESP32 and Cortex-M, on PCs and the Raspberry Pi, and as WebAssembly in the browser, each with dedicated SIMD optimizations from AVX-512 and NEON to the ESP32's PIE instructions. Sample-accurate timing, deterministic results on every platform, and a single test suite that runs in emulators and on real hardware make it a dependable foundation for firmware, desktop tools, web apps and AI-assisted signal analysis.

## Name

SDRk (Software-Defined Radio Kernels) is pronounced *shark*. Text, code comments and documentation use **SDRk**; in code (identifiers, crate and file names, string literals) it is lowercase `sdrk`.

## Scope

The repository contains only reusable SDR building blocks and what is needed to verify them. Applications, UIs, demos, services and device drivers live in their own repositories.

## Status

Milestones M00 to M15, M19 to M25, M24.1 and M25.1 are delivered: the kernels, the blocks, the JSON pipeline with time, events and the transmit path, file I/O and device capabilities, the SIMD backend (x86-64, AArch64, wasm32), the WebAssembly bindings with the npm package layout, the emulator targets (Cortex-M4, Cortex-M55, RISC-V 32, ESP32-P4, ESP32-S3), NFM receive with a squelch (M19), AM receive with an AGC (M20), SSB receive (M21), audio filters (M22), real-sampling devices (M23), FM stereo (M24), the block names (M24.1), FM transmit audio (M25) and the release 0.1.0 (M25.1, [release notes](docs/releases/0.1.0.md)). M16 to M18 are planned. The workspace version is 0.1.0, released as the tag `v0.1.0`; nothing is published to crates.io or npm. See [docs/milestones/README.md](docs/milestones/README.md) and [docs/open-items.md](docs/open-items.md).

## Documentation

- Project plan: [docs/PLAN.md](docs/PLAN.md)
- Milestones: [docs/milestones/README.md](docs/milestones/README.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- Device API and conformance checks: [docs/device.md](docs/device.md); file I/O: [docs/io.md](docs/io.md); virtual devices, loopback channel and system tests: [docs/testing.md](docs/testing.md)
- Backends and the SIMD levels: [docs/backends.md](docs/backends.md); instruction counts against VOLK and rustfft: [docs/performance.md](docs/performance.md)
- WebAssembly bindings and the npm package: [docs/wasm.md](docs/wasm.md)
- Command-line examples: [examples/README.md](examples/README.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo xtask check
cargo xtask bare-metal
cargo xtask schema --check
```

The WebAssembly package is built with `cargo xtask wasm-pkg` (needs `wasm-bindgen-cli`, see [docs/wasm.md](docs/wasm.md)).

`cargo xtask test-all` runs the first five, the WebAssembly package tests when Node.js is installed and the Cortex-M4 emulator run. `cargo xtask emu --tier 1|2` runs the test suite bare-metal in emulators (Cortex-M4/M55, RISC-V, ESP32-P4, ESP32-S3; needs QEMU, see [docs/embedded.md](docs/embedded.md)). `cargo xtask count --platform x86_64` measures the instructions per sample of the SIMD kernels against VOLK and rustfft (needs valgrind and `libvolk-dev`, see [docs/performance.md](docs/performance.md)); `RUSTFLAGS="--cfg sdrk_fallback_only"` makes the facade use the portable fallback everywhere. `cargo xtask schema` regenerates `schema/sdrk-pipeline-v1.json` after a change of a block description; the pipeline model and its JSON format are described in [docs/pipeline.md](docs/pipeline.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
