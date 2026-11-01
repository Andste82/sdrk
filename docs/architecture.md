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
8. Bindings configure, Rust processes. The WebAssembly bindings (`bindings/wasm`) are the first: see [wasm.md](wasm.md).

See [PLAN.md](PLAN.md) for the full list and the reasoning.

## Extensibility rules

- The node group is the single extension point (threads, MCU cores, GPU and network splits); kernels, backends and blocks stay untouched.
- No global mutable state.
- Everything is `Send`.
- Edges are interfaces.
- Determinism is independent of partitioning.

## Partitioning

A **node group** is a set of nodes that run together. The model has groups today (`groups`, `nodes.<name>.group`, see [pipeline.md](pipeline.md#groups-and-backend-preferences)); they have no runtime effect yet. `split_by_groups` turns a description into one ordinary description per group. Every edge between nodes of different groups becomes an output of the producer part and an input of the consumer part, with the type and the exact rate. The parts are normal pipelines: they validate, build and run like any other, and the **edges are the interfaces**. What crosses an edge is described by the [wire format](wire-format.md): a hello with the stream description, samples with their start index, and events (gaps, overflows, anchors) in band. Because the index mapping and the anchor arithmetic are exact node by node, and because every block is block-size invariant, a split pipeline gives bit-identical results and the same events as the single pipeline. The integration test `crates/pipeline/sdrk-pipeline/tests/probe_two_threads.rs` proves this for the FM receiver: group `front` in one thread, group `audio` in another, connected by `std::sync::mpsc` carrying wire-encoded messages cut at random places, ten random chunk plans, anchors and flags included.

The paper check below shows that two more kinds of partitioning need no change of kernels, blocks or the model. Nothing of this is code.

**A GPU group.** One group is mapped to a device queue. The runner of that group is a different implementation of the same contract: it takes the nodes of the group as data (the kind, the parameters and the exact rates are in the description), compiles or selects device kernels for them (the kernel names are those of `BlockFactory::kernels`, and the backend preference `{"require": "gpu"}` already expresses the wish), and keeps the group's state in device memory. Its edges are buffers: the input edge is an upload (the wire samples message is the layout of the buffer: start index, count, element type), the output edge a download. The hello of an edge stream describes the buffer (type, rate), the start indices tell the runner where a gap is, and the events that cross an edge in band are applied on the host side of the edge. The mapping rule needs only `L/M`, the delay and the origin of each node, all of which the group's description provides; `map_index` of a node stays a property of the block. Determinism is a requirement on the device kernels (they must match the fallback within the declared class of the kernel); the pipeline above does not change.

**A network split.** The two parts run on different machines. The edge is a byte stream (TCP) or a message stream (WebSocket): the sender encodes the hello, then the samples and events with `wire::encode`, the receiver appends the received bytes to a buffer and calls `wire::decode_stream`, keeping the bytes of an incomplete message for the next read, exactly as the probe does with its randomly cut pieces. Anchors and events travel in band, so the absolute time of a sample is known on both sides without a clock on the receiving machine; timed parameters cross as `param_set` messages and are acknowledged with `param_ack`. A lost connection is a gap: the receiver sees a jump of the start index and the `Gap` event of its own pipeline. Unknown message types are skipped, so a newer sender can add messages without breaking an older receiver. Compression, encryption and authentication belong to the transport and are not part of the format.

In both cases the changes are on the *outside* of the model: a new runner for a group and a new transport for an edge. The description, the blocks, the kernels and the wire format stay as they are.

## Automatic checks

`cargo xtask check` runs `layers`, `heap`, `global-state`, `names` (the block names of `docs/blocks.md`, section "Names") and `docs` (each can be run alone as `cargo xtask check <name>`); `heap` also rejects every use of `std` in the core layer but the two square roots of `sdrk_core::num` behind the feature `std` (D67: a new one needs a maintainer decision); `cargo xtask bare-metal` builds every heap-free crate for `thumbv7em-none-eabihf`. `cargo xtask schema` writes the JSON Schema of the pipeline description from the block descriptions (`schema/sdrk-pipeline-v1.json`); `cargo xtask schema --check` fails if the committed file is out of date. `cargo xtask count` measures instructions per sample against the comparison libraries and judges them with `benches/limits.json` ([performance.md](performance.md)); `cargo xtask wasm-pkg` and `cargo xtask wasm-test` build and test the WebAssembly package ([wasm.md](wasm.md)).
