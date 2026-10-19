# Open items

Items that remain open, are not covered by a later milestone, and need a decision or input that cannot be taken autonomously. Format: milestone, item, why it is open, what decision or input is needed.

| Milestone | Item | Why it is open | Needed decision / input |
| --- | --- | --- | --- |
| M10 | The nightly `fuzz` job (pipeline_json) has not run | it is skipped on push and pull_request runs; locally a 60 second fuzz run passes | Confirm that the first nightly `fuzz` run finds nothing. |
| M12 | The nightly `system-long` job (ten minute run) has not run | it is skipped on push and pull_request runs; locally the ten-minute test passes in release mode in 15 seconds | Confirm that the first nightly `system-long` run passes. |
| M13 | The nightly job `count-nightly` (AVX-512 through SDE, AArch64 through QEMU, wasm32 through wasmtime) has not run | it is skipped on push and pull_request runs; `avx512-sde`, `aarch64-qemu` and `wasm-wasmtime` are green on main; locally `xtask count` passes for all four platforms | Confirm the first nightly run; check that the SDE download link of Intel stays reachable (the checksum is pinned). |
| M14 | M14-1: the npm package is not published, and the name `sdrk-wasm` is not reserved | `npm view sdrk-wasm` answered 404 on 2026-10-19; publishing is a maintainer decision (decided on 2026-10-19: decide later) | Decide the package name (unscoped or scoped), who publishes, and when; re-check the name right before the first `npm publish`. |
| M15 | M15-1: the CI jobs of M15 (`emu-tier1`, the nightly `emu-tier2` matrix, `count-embedded`) have not run | only a local Linux x86-64 container without GitHub Actions was available; the runner image ships QEMU 8.2 where the development machine has 10.2.1 | Confirm that the first CI run on the M15 branch is green; if the runner's QEMU is too old for a machine, install a pinned QEMU as `count-embedded` does. |

## Decided on 2026-10-19, implementation pending

These were open items; the decision is taken (see `docs/decisions-log.md`), the work is not done yet.

| Origin | Decision | Work |
| --- | --- | --- |
| M10 | A real converter block is added | `convert_out_real` block in `sdrk-blocks` with a compatible schema entry |
| M13-1 | `convert_out` rounds half to even | change the numeric contract of M08 (fallback, backend, golden vectors, `docs/kernels.md`), drop the waiver |
| M13-2 | `FftPlan` may carry per-stage twiddle tables and scratch | change the M07 plan API and `docs/kernels.md`, optimise the FFT, set a realistic limit |
| M13-3 | a true radix-4 butterfly | more accurate FFT at N = 64 on wasm32, drop the waiver |
| M13-6 | `neon` is counted natively on an arm64 runner | CI job on `ubuntu-24.04-arm` with callgrind and VOLK for aarch64 |
| M13-7 | `sdrk-blocks` forwards `std` | feature `std = ["sdrk/std"]` |
| M15-3 | cargo feature `hardware` for the ESP32-S3 target | call `esp_hal::init` with the feature, the manual set-up without it |
| M15-4 | ESP32-P4 quick profile nightly, full profile weekly | change the schedule of `emu-tier2` |
