# SDRk

A Rust library of SDR building blocks for receiving and transmitting, from mixers and filters to complete demodulation chains, composed as JSON pipelines and controllable at runtime. The same kernels run heap-free on microcontrollers like the ESP32 and Cortex-M, on PCs and the Raspberry Pi, and as WebAssembly in the browser, each with dedicated SIMD optimizations from AVX-512 and NEON to the ESP32's PIE instructions. Sample-accurate timing, deterministic results on every platform, and a single test suite that runs in emulators and on real hardware make it a dependable foundation for firmware, desktop tools, web apps and AI-assisted signal analysis.

## Name

SDRk (Software-Defined Radio Kernels) is pronounced *shark*. Text, code comments and documentation use **SDRk**; in code (identifiers, crate and file names, string literals) it is lowercase `sdrk`.

## Scope

The repository contains only reusable SDR building blocks and what is needed to verify them. Applications, UIs, demos, services and device drivers live in their own repositories.

## Status

Foundation stage: the workspace, the layer rules and their automatic checks exist; the libraries are filled in milestone by milestone.

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

`cargo xtask test-all` runs the first five. `cargo xtask count --platform x86_64` measures the instructions per sample of the SIMD kernels against VOLK and rustfft (needs valgrind and `libvolk-dev`, see [docs/performance.md](docs/performance.md)); `RUSTFLAGS="--cfg sdrk_fallback_only"` makes the facade use the portable fallback everywhere. `cargo xtask schema` regenerates `schema/sdrk-pipeline-v1.json` after a change of a block description; the pipeline model and its JSON format are described in [docs/pipeline.md](docs/pipeline.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
