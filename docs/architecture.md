# Architecture

## Layers

The folder of a crate is its layer. A crate may depend only on crates in lower layers; `cargo xtask check layers` enforces this.

| Layer (folder) | Crates | May depend on | `no_std`, no heap |
| --- | --- | --- | --- |
| `crates/core` | `sdrk-device` (rank 0), `sdrk-core` (rank 1) | lower rank | yes |
| `crates/backends` | `sdrk-backend-*` | `sdrk-core`, `sdrk-device` | yes |
| `crates/facade` | `sdrk` | core, backends | yes |
| `crates/blocks` | `sdrk-blocks` | facade, core | yes |
| `crates/pipeline` | `sdrk-pipeline` | blocks, facade, core | no |
| `crates/io` | `sdrk-io` | core (types only) | no |
| `crates/testing` | `sdrk-testkit` | any library crate | `no_std` by default, feature `std` |
| `bindings/*`, `examples/*`, `targets/*`, `benches/*`, `xtask` | — | anything | no |

## Principles

1. Kernels, not a SIMD abstraction: a reference, a fallback and optional protokernels per backend.
2. Four internal sample types: `C32`, `CQ15`, `f32`, `Q15`; conversion only at the edges.
3. Everything up to and including `blocks` is `no_std` and heap-free; states are generic over their storage.
4. Traits per sample type; the facade composes per kernel which backend runs it.
5. Receive and transmit are equal citizens.
6. The sample index is the time base; the pipeline never reads a clock.
7. Deterministic: same backend, bit-identical output regardless of block size or partitioning.
8. Bindings configure, Rust processes.

See [PLAN.md](PLAN.md) for the full list and the reasoning.

## Extensibility rules

- The node group is the single extension point (threads, MCU cores, GPU and network splits); kernels, backends and blocks stay untouched.
- No global mutable state.
- Everything is `Send`.
- Edges are interfaces.
- Determinism is independent of partitioning.

## Automatic checks

`cargo xtask check` runs `layers`, `heap`, `global-state` and `docs`; `cargo xtask bare-metal` builds every heap-free crate for `thumbv7em-none-eabihf`. `cargo xtask schema` writes the JSON Schema of the pipeline description from the block descriptions (`schema/sdrk-pipeline-v1.json`); `cargo xtask schema --check` fails if the committed file is out of date.
