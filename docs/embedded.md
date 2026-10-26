# Embedded test targets

The heap-free promise is only real if the code runs on microcontrollers. The testkit suite (`sdrk_testkit::suite`, see [testing.md](testing.md)) is `no_std` and works on a caller-provided arena, so the same cases that run on the host with `cargo test` run bare-metal in emulators. Everything is driven by `cargo xtask emu`; `cargo xtask test-all` includes the tier 1 target.

## Targets

| Directory (`targets/`) | Tier | Rust target | Emulator and machine | Output | Exit |
| --- | --- | --- | --- | --- | --- |
| `qemu-cortex-m4` | 1 | `thumbv7em-none-eabihf` | `qemu-system-arm -machine mps2-an386 -cpu cortex-m4` | semihosting | semihosting exit |
| `qemu-cortex-m55` | 2 | `thumbv8m.main-none-eabihf` | `qemu-system-arm -machine mps3-an547` | semihosting | semihosting exit |
| `qemu-riscv32` | 2 | `riscv32imac-unknown-none-elf` (scalar fallback, soft float) | `qemu-system-riscv32 -machine virt -bios none` | semihosting | semihosting exit |
| `esp32p4` | 2 | `riscv32imafc-unknown-none-elf` | `esp-emu --chip esp32p4` | UART0 | the host stops the emulator after `SDRK-SUITE END` |
| `esp32s3` | 2 | `xtensa-esp32s3-none-elf` | Espressif QEMU fork, `qemu-system-xtensa -machine esp32s3` | UART0 | the host stops the emulator after `SDRK-SUITE END` |

Tier 1 runs on every push and pull request (quick profile), tier 2 nightly (quick and full; the full profile of the ESP32-P4 weekly). All five targets pass both profiles on Linux x86-64.

### Structure

`targets/` is excluded from the main workspace (`exclude = ["targets/*", ...]` in the root `Cargo.toml`). Every target is a small Cargo project of its own with `.cargo/config.toml` (target triple, linker arguments), its own `Cargo.lock` and the build profile `emu` (`inherits = "release"`, `overflow-checks`, `debug-assertions`, `lto`, `codegen-units = 1`, `panic = "abort"`). `cargo test --all-features` of the workspace never sees them.

- `targets/common` (`sdrk-target-common`) holds what is the same everywhere: the arena size (`ARENA_SIZE`, 256 KiB), the tier switch (feature `full`, default quick), `run()` (the testkit's `run_suite` with a `WriteReporter` over any `core::fmt::Write`) and `report_panic()` (the `SDRK-PANIC <message>` line).
- Each target has its own `main`: it sets up the platform minimally (no heap, no allocator), places the 256 KiB `static` arena **in the target crate** (targets are applications; R6 applies to library crates), runs `run_suite(all_cases(), tier, ...)`, prints the protocol lines and signals the exit status. The panic handler prints `SDRK-PANIC <message>` and exits with failure; any panic is a bug.
- `qemu-cortex-m4` has the feature `inject-failure` (one more, always failing case defined in the target) and a second binary, `count` (see "Instruction counts").

### The protocol

The suite prints (see the module documentation of `sdrk_testkit::suite`)

```text
SDRK-SUITE START <n>
SDRK-CASE PASS <name>
SDRK-CASE FAIL <name> <message> <detail>
SDRK-SUITE END passed=<p> failed=<f>
SDRK-PANIC <message>
```

The host decides pass or fail from these lines only (`xtask::emu::judge`; the exit code of the emulator is never used, and `esp-emu` returns 0 on a match regardless of failures). A run passes when exactly one `START` and one `END` are present, `failed=0`, there is no `FAIL` and no `PANIC` line, `START` announced `passed + failed` cases and the number of `PASS` lines equals `passed` (a lost or garbled line is a failure, not a silent pass). A console prefix or ANSI colour codes in front of a marker are ignored.

### Memory

Every case works in a 256 KiB arena (quick cases in 64 KiB); a case that needs more fails with `CapacityExceeded`, reported as `FAIL`. Two host tests catch oversizing before the emulator: `full_suite_passes_with_a_256_kib_arena` and `quick_suite_passes_with_a_64_kib_arena` (`crates/testing/sdrk-testkit/tests/suite.rs`).

The suite on the MCUs is the whole testkit suite: kernels in both directions, fused kernels, blocks with `ArenaStorage` and the end-to-end FM chains (`e2e_fm_rx_*`, `e2e_fm_txrx_*`; 0.1 s in the quick profile, 1 s in the full profile), NFM (`e2e_nfm_rx_*`; 2 s variants in the full profile) AM (`e2e_am_rx_*`; 0.2 s quick, 1 s full) and SSB (`e2e_ssb_rx_*`; 0.2 s; `e2e_ssb_rx_q15` with the upper sideband in the quick profile, the `f32` chain needs about 75 KiB and runs in the full profile). The `C32` cases run with the hardware FPU on the Cortex-M4F, Cortex-M55, ESP32-P4 and ESP32-S3 and in soft float on RISC-V `imac`. The `f64` of the references is software floating point on every target but the Cortex-M55 (the P4 and the M4F have single-precision FPUs only), which dominates the run times.

## Memory maps and linker scripts

| Target | Code | Data and stack | Source |
| --- | --- | --- | --- |
| `qemu-cortex-m4` | 4 MiB SRAM at `0x00000000` | 4 MiB SRAM at `0x20000000` | QEMU documentation "Arm MPS2 and MPS3 boards"; `hw/arm/mps2.c` (AN386) |
| `qemu-cortex-m55` | 512 KiB ITCM at `0x00000000` (the reset vectors are read from address 0) | 4 MiB SRAM at `0x21000000` | QEMU 10.2.1 `info mtree` of `mps3-an547` (secure alias); `hw/arm/mps2-tz.c` |
| `qemu-riscv32` | RAM at `0x80000000` (16 MiB used of the default 128 MiB) | same RAM, stack at the top | QEMU documentation "RISC-V Generic Virtual Platform"; `hw/riscv/virt.c` |
| `esp32p4`, `esp32s3` | the linker scripts of `esp-hal` (`linkall.x`) | | `esp-hal` 1.2.2 |

`memory.x` is copied by `build.rs` into `OUT_DIR` so that `cortex-m-rt` (0.7) and `riscv-rt` (0.18) find it. The Cortex-M55 needs the optimisation level `s`: the code (about 400 KiB) must fit the ITCM, which is where the vector table is.

## ESP32 builds

`esp-hal` 1.2.2 (pinned with `=`) supports the ESP32-P4 and the ESP32-S3, so the ESP-IDF fallback of the milestone (a C application linking the suite as a `staticlib`, built in the official Docker image) was **not needed** and is not implemented. Output goes through `esp-println` (feature `uart`, UART0). The flash image is made by `espflash save-image --chip <chip> --merge --flash-size 16mb`, which adds the ESP-IDF second stage bootloader and the partition table.

- **ESP32-P4** (`esp-emu`): the target calls `esp_hal::init`. The emulator interprets the code (soft-float `f64` in the references is slow there): the quick profile takes 18 to 28 minutes and the full profile about two hours (the 4096-point DFT of the reference alone takes more than an hour), so the timeouts of this target are 3600 s (quick) and 14400 s (full).
- **ESP32-S3** (Espressif QEMU fork): the target does **not** call `esp_hal::init`. `init` switches the CPU to the 480 MHz PLL and spins on the analog lock of the PLL, which the emulator does not model, so it hangs. The target keeps the startup code and the linker script of `esp-hal` and does by hand what the program needs: the timer-group, RTC and super watchdogs that the bootloader leaves running are switched off (register addresses from the `esp32s3` peripheral access crate), and the floating-point coprocessor is enabled (`CPENABLE` bit 0; otherwise the first `f32` instruction raises an exception that the runtime cannot handle in this state). With the cargo feature `hardware` the target is the variant for real boards: `main` calls `esp_hal::init(Config::default())` (CPU clock, interrupts, timers) and skips the manual set-up. The default stays the emulator set-up, so `cargo xtask emu --target esp32s3` is unchanged. The `hardware` build cannot run in the emulator (`init` hangs on the PLL lock); it is only compiled: `cargo xtask emu --target esp32s3 --build-only --features hardware` (and without the feature for the default), run by the CI job `esp32s3-hardware-build`. `--build-only` builds the selected targets and stops, without image or emulator. It has not run on a board (open item M15-3). Link-time LTO is off for this target: `lto = "fat"` breaks the literal pool placement of the Xtensa linker (`dangerous relocation: l32r`).
- The Xtensa Rust toolchain is the `esp` toolchain of `espup`, pinned to **1.95.0.0**: with 1.99.0.0 `esp-hal` 1.2.2 does not link (`dangerous relocation: l32r: literal placed after use` in `__pre_init`, also in a minimal program without SDRk code). The target has its own `rust-toolchain.toml` (`channel = "esp"`), and `esp-hal` needs Rust 1.95, so both ESP targets declare `rust-version = "1.95"`.
- **Version table** of what was used and verified: `esp-hal` 1.2.2, `esp-println` 0.18.0, `esp-bootloader-esp-idf` 0.6.0, `espflash` 4.6.0, `espup` 0.17.1 with Rust 1.95.0.0 (Xtensa GCC `esp-15.2.0_20250920`), `esp-emu` 0.49.0, Espressif QEMU `esp-develop-9.2.2-20260417` (QEMU 9.2.2).

## `cargo xtask emu`

```text
cargo xtask emu [--target <name>]... [--tier 1|2] [--all] [--full]
                [--features a,b] [--timeout <s>] [--skip-missing] [--install] [--list]
```

- Selection: `--target` (repeatable), `--tier 1` or `--tier 2`, `--all`; without a selector the tier 1 target runs. `--list` shows the table.
- For every target it builds with the profile `emu` (`cargo build --profile emu --bin <binary> --target-dir target/emu/build/<name>` in the target directory, with `RUSTUP_TOOLCHAIN` removed so that the target's own `rust-toolchain.toml` applies), makes the image (the ESP targets), runs the emulator with a timeout, streams the output to the console and to `target/emu/<name>.log`, and reads the protocol. The emulator is stopped as soon as `SDRK-SUITE END` has been seen (the ESP emulators do not stop by themselves).
- Profiles: quick (default) and `--full` (adds the cargo feature `full`). Timeouts: quick 300 s and full 3600 s, longer where measured to be needed (Cortex-M4 600 s quick, Cortex-M55 900 s quick, RISC-V 900 s quick, ESP32-P4 3600 s / 14400 s, ESP32-S3 900 s / 7200 s); `--timeout <s>` overrides all of them.
- `--features` is passed to the target crate (for example `--features inject-failure`). The summary table lists result, passed, failed and seconds per target. The exit code is 1 on any failure (also a timeout, a missing `END`, a build failure) and 2 on bad arguments.
- A missing tool is a failure with an install hint; `--skip-missing` turns it into `SKIP` (used where esp-emu does not exist, e.g. Windows).
- `--install` downloads the pinned tools into `target/tools/`: `esp-emu` and the Espressif QEMU fork (release archives, **SHA-256 verified** before unpacking, no `curl | sh`), `espflash` and `espup` (`cargo install --locked`, pinned version) and, through `espup`, the Xtensa Rust toolchain. Arm and RISC-V QEMU come from the distribution (`apt-get install qemu-system-arm qemu-system-misc`; Ubuntu 26.04 splits the RISC-V machines into `qemu-system-riscv`). The locations can be overridden with `SDRK_QEMU_ARM`, `SDRK_QEMU_RISCV32`, `SDRK_ESP_EMU`, `SDRK_ESP_QEMU_XTENSA`, `SDRK_ESPFLASH` and `SDRK_XTENSA_GCC`.
- The pins are in `xtask/src/emu.rs` (`pins`): `esp-emu` 0.49.0, Espressif QEMU `esp-develop-9.2.2-20260417`, `espflash` 4.6.0, `espup` 0.17.1, Rust 1.95.0.0.

`cargo xtask test-all` runs the commands of R1.2, the tests of the WebAssembly package when Node.js is installed (`wasm-pkg` and `node --test`) and `emu --tier 1`.

The parser and the process handling are tested on the host (`xtask/tests/emu.rs`): a log with a `FAIL`, with a missing `END`, with a `PANIC` and with `failed=1` is a failure, a log with only `PASS` lines and a clean `END` passes, a process that keeps running after `END` is stopped, a silent process times out, and a zero exit code never rescues a failing log. The target crate `qemu-cortex-m4` with `--features inject-failure` proves the whole chain: the CI job `emu-tier1` runs it and expects exit code 1.

## Instruction counts (Cortex-M4)

```text
cargo xtask count --platform cortex-m4 [--kernel NAME]... [--no-write]
```

runs fifteen kernels of the portable fallback (`convert_cs8`, `convert_cs16`, `convert_out_cs16`, `rotate`, `fir`, `fir_decimate`, `mag_sq`, `quad_demod`, `fm_mod`, `fft_1024`, `fft_cq15_1024`, and since M20 `agc`, `agc_cq15`, `dc_block_real`, `dc_block_real_q15` with the inputs and parameters of `benches/kernels`) under `qemu-system-arm` with the `insn` plugin and writes `benches/results/cortex-m4.json` with the instructions per sample. The method is the one of the other platforms ([performance.md](performance.md)): the second binary of the target crate, `count`, takes `<kernel> <k>` from the semihosting command line, runs the kernel once as a warm-up and then `k` times (each kernel in a function of its own that is never inlined, so that adding a kernel to the binary does not change the code generated for the others); the count of `k = 16` minus the count of `k = 0`, divided by the samples of the 16 calls (1024 per call), is the figure. The plugin is built as in M13 (`libinsn.so` from the QEMU tag `v10.2.1`, `SDRK_QEMU_TAG`/`SDRK_QEMU_SRC`; the QEMU binary must be built with plugin support, `SDRK_QEMU_ARM`). There are no limits yet: the comparison with CMSIS-DSP comes with the `arm-dsp` backend (backlog).

## Running on real boards

The same `targets/` will run on real boards through probe-rs (backlog). The ESP32-S3 target uses its cargo feature `hardware` for that (see above).
