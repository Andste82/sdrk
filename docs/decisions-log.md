# Decisions log

One line per judgement call: date, milestone, decision, reason.

| Date | Milestone | Decision | Reason |
| --- | --- | --- | --- |
| 2026-10-18 | M00 | `xtask` is a library plus a thin binary | the checks can be tested against fake workspaces in `xtask/tests` |
| 2026-10-18 | M00 | workspace dependencies: `num-complex` 0.4.6, `libm` 0.2.16, `bytemuck` 1.25.2, `heapless` 0.9.3; `xtask` uses `anyhow`, `cargo_metadata`, `walkdir`, `tempfile` | newest releases at implementation time; `xtask` is a host tool outside the heap-free layers |
| 2026-10-18 | M00 | the global-state check exempts `#[cfg(test)]` code | test helpers may use statics; library code may not |
| 2026-10-18 | M00 | the heap check treats out-of-line modules declared under `#[cfg(test)]` or `#[cfg(feature = "std")]` like inline ones | a `mod tests;` file must follow the exemption of its declaration |
| 2026-10-18 | M00 | `cargo xtask bare-metal` could not be run during implementation | the cloud session cannot download the `thumbv7em-none-eabihf` target; later verified locally in the development container and in CI |
| 2026-10-18 | M00 | CI installs the toolchain explicitly with `rustup toolchain install` (no arguments, so it reads `rust-toolchain.toml`) | rustup 1.28 no longer installs the toolchain from `rust-toolchain.toml` automatically; one source of truth for channel, components and targets |
| 2026-10-18 | M00 | `deny.toml` keeps the licence allowlist of the milestone although several entries are not used yet | `cargo deny` only warns about unused allowances; later crates will need them |
| 2026-10-18 | M00 | the arm64 CI entry uses the GitHub-hosted `ubuntu-24.04-arm` runner | if the repository plan lacks it, switch to a self-hosted runner labelled `linux-arm64` |
| 2026-10-18 | M00 | `.devcontainer/` (Dockerfile, devcontainer.json) based on Ubuntu 26.04 with Rust (stable + nightly/Miri), all targets of M00-M16, QEMU, valgrind, libvolk, binaryen, Python, Node, wasm-bindgen-cli, wasmtime, cbindgen, cargo-deny, iai-callgrind-runner, Claude Code and gh | one reproducible environment for all milestones; not built or tested in the cloud session |
| 2026-10-18 | M00 | Intel SDE, esp-emulator, Espressif QEMU, espup/ESP-IDF and browsers are not in the image | the milestones download them pinned and checksum-verified via xtask or CI actions |
| 2026-10-18 | M00 | xtask tests use `assert_eq!`/`assert_ne!` against an empty collection instead of `assert!(x.is_empty())` | `clippy::assert_is_empty` (new in stable clippy) failed CI; the new form also prints the findings on failure |
