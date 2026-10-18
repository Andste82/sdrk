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
| 2026-10-18 | M00 | `.devcontainer/` (Dockerfile, devcontainer.json) based on Ubuntu 26.04 with Rust (stable + nightly/Miri), all targets of M00-M16, QEMU, valgrind, libvolk, binaryen, Python, Node, wasm-bindgen-cli, wasmtime, cbindgen, cargo-deny, iai-callgrind-runner, a coding assistant CLI and gh | one reproducible environment for all milestones; not built or tested in the cloud session |
| 2026-10-18 | M00 | Intel SDE, esp-emulator, Espressif QEMU, espup/ESP-IDF and browsers are not in the image | the milestones download them pinned and checksum-verified via xtask or CI actions |
| 2026-10-18 | M00 | xtask tests use `assert_eq!`/`assert_ne!` against an empty collection instead of `assert!(x.is_empty())` | `clippy::assert_is_empty` (new in stable clippy) failed CI; the new form also prints the findings on failure |
| 2026-10-18 | M00 | the container runs as root (no `ubuntu` user switch); `devcontainer.json` uses `remoteUser: root` | toolchains in `/usr/local` stay writable for `rustup target add` and `cargo install`; `start.sh` mounts the assistant config under `/root` |
| 2026-10-18 | M00 | `.devcontainer/start.sh` (bash) starts or attaches to the container on Linux hosts | developer convenience outside build and test; R1.3 (no shell scripts) covers the build, test and `xtask` tooling, which stays pure Rust |
| 2026-10-18 | M01 | `Rate::hz(u64)` returns `Result<Rate>` | it is shorthand for `new(hz, 1)` and must keep the invariant `num > 0` (0 Hz would divide by zero in `time_of`) |
| 2026-10-18 | M01 | `SampleFormat::new` also rejects `bits == 0`, and `with_full_scale` is the only fallible builder | a zero-bit component is meaningless; `with_swap_iq` cannot fail |
| 2026-10-18 | M01 | `ErrorKind::as_str`, getters `num()`/`den()` on `Rate` and `Ratio`, and `BlockFlags::ALL` (private) were added | needed for `Display` and for callers; they extend the specified API without changing it |
| 2026-10-18 | M01 | `Ratio::mul` carries `#[allow(clippy::should_implement_trait)]` | the name is normative and the operation is fallible, so `std::ops::Mul` does not fit |
| 2026-10-18 | M01 | `BlockFlags::from_bits_truncate` drops bits outside the seven defined flags | keeps unknown bits from leaking into `contains` |
| 2026-10-18 | M01 | tests live in `crates/core/sdrk-device/tests/device.rs` (integration) instead of unit modules | the crate is `no_std`; integration tests have `std` for `to_string` |
| 2026-10-18 | M01 | `full_scale_value` for `FullScale::PowerOfTwo` on `U8` returns 128 | `2^(bits-1)` with bits = 8, matching the spec |
| 2026-10-18 | M01 | `samples_in_bytes` saturates instead of failing for very large byte counts | its signature returns `usize`, not `Result` |
| 2026-10-18 | M02 | `Delay::add` returns `Result<Delay>` and reduces its result; `Delay` equality compares the stored fraction, so `reduced()` is needed before comparing different spellings | the spec lists `add()` without a signature; overflow must not panic (R5), and the derived `PartialEq` is specified |
| 2026-10-18 | M02 | `Delay` has getters `num()`/`den()`; `Delay::new` rejects a zero denominator with `Error::InvalidParameter`; `add` carries `#[allow(clippy::should_implement_trait)]` | same reasoning as `Ratio::mul` in M01: the operation is fallible |
| 2026-10-18 | M02 | `Acc64` is a tuple struct with a public field and `mac(&mut self, ..)` saturates its additions; `round_shift` uses a saturating add and returns the sign for shifts of 63 or more | R5 forbids unchecked overflow; the shift guard keeps `1 << (s - 1)` defined for every input |
| 2026-10-18 | M02 | `q15_from_f64` is public in `num` (used by `Q15::from_f64_round`); `Q15::from_f32` delegates to `q15_from_f32` | one implementation of the R7 conversion |
| 2026-10-18 | M02 | `delegate!` accepts several entries separated by `;` and must be invoked exactly once; it also emits `backend_for` and its table | a macro_rules macro cannot collect entries from separate invocations, so one invocation is the single source of truth for impls and table |
| 2026-10-18 | M02 | backend names come from a facade trait `BackendName` (`Fallback` -> `"fallback"`) | `stringify!` cannot lowercase; the name stays next to the facade, which is the only crate that knows the backends |
| 2026-10-18 | M02 | the facade re-exports `kernels`, `num` and the core types; the `delegate!` call site must have the trait, sample type and `Result` in scope | keeps `Sdrk` usable with a single dependency; macro hygiene does not resolve names for the caller |
| 2026-10-18 | M02 | `map_kernel!` takes `(input, output, closure)` and returns from the enclosing function on a length mismatch | the spec gives the shape `$expr(i)`; a closure keeps captured parameters such as the gain explicit |
| 2026-10-18 | M02 | `Reference` `Scale<CQ15>` re-implements the integer formula inline instead of calling `num::q15_mul` | the reference is an independent executable specification |
| 2026-10-18 | M02 | tests live in `crates/core/sdrk-core/tests/core.rs` and `crates/facade/sdrk/tests/facade.rs`; the reference tests are gated by `cfg(feature = "reference")` | R8.7 and `no_std` crates; `cargo test -p sdrk-core` without the feature must still pass |
