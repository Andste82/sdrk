# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crates follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- M00: workspace with the layer folders, placeholder crates `sdrk-device`, `sdrk-core`, `sdrk` and `sdrk-testkit`, `xtask` (`check`, `bare-metal`, `test-all`), lint and format configuration, `deny.toml`, CI on Linux x86_64, Linux arm64 and Windows x86_64, documentation skeleton, development container (`.devcontainer/`).
- M01: `sdrk-device` with `ErrorKind`/`SdrkError`/`Error`, `SampleFormat` and the `formats` constants, exact time types (`SampleIndex`, `Rate`, `Ratio`, `TimeAnchor`), `BlockFlags`/`BlockMeta` and the `RxStream`/`TxStream` traits; `docs/device.md`.
- M02: `sdrk-core` with the sample types (`C32`, `CQ15`, `F32`, `Q15`), Q15 arithmetic (`num`), `Error`, timing declarations (`Delay`, `Timing`, `Progress`), the one-trait-per-kernel pattern with `Fallback` and `Reference` (feature `reference`), `map_kernel!` and the example kernel `Scale`; facade `sdrk` with `Sdrk`, `delegate!` and `backend_for`; `docs/kernels.md`.
