# Embedded test targets

Each directory is a small Cargo project of its own (outside the main workspace) that runs the
`sdrk-testkit` suite bare-metal in an emulator:

| Directory | Tier | Emulator |
| --- | --- | --- |
| `qemu-cortex-m4` | 1 | QEMU `mps2-an386` |
| `qemu-cortex-m55` | 2 | QEMU `mps3-an547` |
| `qemu-riscv32` | 2 | QEMU `virt` (rv32imac) |
| `esp32p4` | 2 | `esp-emu` |
| `esp32s3` | 2 | Espressif QEMU fork |
| `common` | | shared suite driver (`sdrk-target-common`), not a target |

Run them with `cargo xtask emu` (see [`docs/embedded.md`](../docs/embedded.md)); do not run `cargo build` in
the directories by hand unless you pass `--profile emu` and have the tools and toolchains that
`cargo xtask emu --install` sets up.
