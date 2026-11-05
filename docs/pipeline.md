# Pipeline

The crate `sdrk-pipeline` (layer `crates/pipeline`, heap allowed) turns a description of a signal-processing graph into a running pipeline. The description is data: JSON is the canonical format, so host languages and AI agents never write code to describe processing. The pipeline is the only place with heap: it allocates once while building and then runs the heap-free blocks of [`sdrk-blocks`](blocks.md) deterministically. **Nothing allocates after `build()`** (tested with a counting global allocator in `tests/no_alloc.rs`).

This page covers format version 1, the push-mode runner, metadata, events and timed parameters, pull mode for transmit, node groups, YAML input and the pointer to the wire format. The time base is explained in [timing.md](timing.md), the wire format in [wire-format.md](wire-format.md) and the partitioning in [architecture.md](architecture.md).

```text
JSON text ──parse──▶ PipelineSpec ──validate──▶ Report (errors, warnings, figures)
                          │                          │ no errors
                          └─────────build───────────▶ Pipeline ──push / read_* / set_param
```

## Model

A description is one JSON object. Keys are written in the order of the tables below; maps (`inputs`, `nodes`, `outputs`, `params`) are sorted by key. Unknown fields are errors, so typos are never ignored. Every object may carry an optional `description` (string, at most 2000 characters) that is preserved.

```json
{
  "sdrk": 1,
  "name": "fm-receiver",
  "description": "Receive one FM broadcast station and output 48 kHz audio.",
  "settings": { "precision": "f32", "max_chunk": 4096 },
  "inputs":  { "rf": { "format": "cu8", "rate_hz": 2400000 } },
  "nodes":   { "tune": { "kind": "ddc", "params": { "frequency_hz": 600000, "decimation": 10 } } },
  "edges":   [ { "from": "rf", "to": "tune" }, { "from": "tune", "to": "audio" } ],
  "outputs": { "audio": { "type": "stream", "capacity": 16384, "drop": "error" } }
}
```

### Top level

| Field | Type | Meaning |
| --- | --- | --- |
| `sdrk` | integer | Format version. Must be `1`. |
| `name` | string | Name of the pipeline (free text). |
| `description` | string, optional | Free text. |
| `settings` | object, optional | See below; defaults apply when absent. |
| `inputs` | object | Inputs by name. |
| `groups` | object, optional | Node groups by name, see [Groups](#groups-and-backend-preferences). Written only when present. |
| `nodes` | object | Processing nodes by name. |
| `edges` | array | Connections. |
| `outputs` | object | Outputs by name. |

`settings`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `precision` | `"f32"` or `"q15"` | `"f32"` | `"f32"` uses complex `C32` and real `f32`, `"q15"` uses `CQ15` and `Q15`. One precision per pipeline. |
| `max_chunk` | integer 1 to 1 048 576 | 4096 | Largest number of samples accepted per `push` and input. For a raw input the limit in bytes is `max_chunk` times the bytes of one sample, rounded up. |
| `event_queue` | integer 1 to 65 536 | 256 | Capacity of the event queue (see [Metadata and events](#metadata-and-events)). Written only when given. |
| `pending_params` | integer 1 to 4096 | 16 | Pending timed parameter changes per node (see [Timed parameters](#timed-parameters)). Written only when given. |

### Names

Inputs, nodes and outputs share one namespace. A name matches `^[a-z][a-z0-9_]{0,63}$` and is unique. The pipeline inserts automatic nodes (see below) named `<input>__convert` and `<output>__convert_out`; a user name must not collide with them.

### Inputs

An input is either raw bytes in a sample `format` or already converted samples of a `type` (or packets, `"type": "packets"`). Exactly one of the two is given.

| Field | Type | Meaning |
| --- | --- | --- |
| `format` | name or object | Raw sample format (below). An automatic `convert` node follows the input. |
| `type` | `"c32"`, `"cq15"`, `"f32"`, `"q15"`, `"packets"` | Converted samples. The type must fit the precision: `c32` and `f32` with `"f32"`, `cq15` and `q15` with `"q15"`. `"packets"` is a packet input (below), in either precision. |
| `rate_hz` | integer or `"num/den"` | Sample rate in Hz. A positive integer, or an exact fraction written as a string; it is reduced on load (`"96000/2"` becomes `48000`) and written as an integer if it is whole. Required for every sample input (`missing_param` at `inputs.<name>.rate_hz`); a packet input needs it only when an encoder reads it (below). |
| `description` | string, optional | |

**Format names:** `cu8`, `cs8`, `cs16`, `cs12_lsb16`, `sc16_q11`, `cs16_planar`, `packed12`, `packed10`, `i2s32_msb`, `cf32`, `rf32`, `rs16` map to the constants of `sdrk_device::formats`. A custom format is an object `{ "container": "i16", "bits": 12, "align": "lsb", "layout": "interleaved", "endian": "le", "full_scale": "power_of_two" }` with `container` one of `u8 i8 i16 i32 f32 packed12 packed10 bits1`, `align` `lsb`/`msb`, `layout` `interleaved`/`planar`/`real`, `endian` `le`/`be`, `full_scale` `"power_of_two"` or `{"explicit": 2048}` (optional) and an optional `swap_iq` flag. The object is validated by `sdrk-device`.

A raw input must be a streamable format: interleaved complex or real layout, no 1-bit container. Planar formats (`cs16_planar`) are rejected at `inputs.<name>.format`. A **real** format (`rs16`, `rf32`: Airspy R2 and Mini, direct-sampling front ends, sound cards) gets the real variant of the automatic `convert` node (`convert_real`, `ConvertRealBlock`): the converted stream is `f32` (`q15` in a Q15 pipeline), a real stream, which only the real-aware kinds accept (`halfband_decimate`, `real_to_iq`, `biquad`, `dc_block`, `resample`, `fm_mod`, `preemph_limit`, `deemphasis`, `decimate`, `interpolate`, `delay`, `to_complex`, `nco`); put `real_to_iq` behind it to get a complex stream at half the rate (see `real_input_fm_receiver.json`). Real audio that is already converted can still be given as `"type": "f32"`.

**Packet inputs** (M37). `"type": "packets"` takes [`Packet`](blocks.md#packet-ports)s (up to 256 bytes with a sample index and flags) with `Pipeline::push_packets` instead of samples; up to `max_chunk` packets per push. Its `rate_hz` is the sample rate of the stream that a pure encoder (timing kind `Source`, the kind `frame_encode`) produces from it. It is required when an encoder reads the input at port `a` (`missing_param` at `inputs.<name>.rate_hz`, checked in stage 1 from the edges and the kinds) and ignored with the warning `invalid_value` at the same path when only port `b` of clocked encoders reads it (they take their rate from their clock; D514). A packet input without `rate_hz` that only goes to outputs is valid; `Pipeline::input_rate` of such an input is `InvalidState`. A packet input starts its own clock domain (its name), so a stream that an encoder produces from it cannot be mixed with the stream of another input (`clock_domain`, see [Edges](#edges)); two encoders of the same packet input are one clock. Port `a` of such an encoder reads only a packet input: the packets of a decoder have no sample rate (their nominal rate only sizes buffers, D513) and the clock of the decoder's input, so a decoder feeding an encoder is `type_mismatch` at the edge (D526); a relay reads the packets out and pushes them into a packet input of its own.

### Nodes

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | string | A block kind of the registry, e.g. `"ddc"`. The kinds, their ports and all parameters are listed in [blocks.md](blocks.md) and in the schema. The kinds `add`, `mul`, `make_complex` and `stereo_pll` have two inputs, see [Edges](#edges). |
| `params` | object, optional | Parameters by name. |
| `group` | string, optional | A key of `groups`; nodes without one are in the implicit group `"main"`. |
| `backend` | object, optional | `{"prefer": name}` or `{"require": name}`, see [Groups](#groups-and-backend-preferences). |
| `align` | string, optional | `"auto"` (default) or `"none"`: what the build does when the two paths into a node with two inputs differ in group delay, see [Edges](#edges). Every node accepts it; it has no effect on a node with one input. |
| `description` | string, optional | |

A parameter value is a JSON number, a boolean or a string. Integers must be exact: `10` and `10.0` are accepted for an integer parameter, `10.5` is not. A `Choice` parameter (for example the `window` of `spectrum`) is given by its option name, a `Format` parameter (`convert`, `convert_out`) by a format name or object. A parameter that is not given takes the default of the description or the rate-dependent default of the block (for example the cutoff of the `ddc`). Parameters listed under `required` in the description (for example `decimation` of `ddc`, `rate_out_hz` of `resample`) must be given.

Parameters are either **build-time** (fixed when the pipeline is built) or **runtime** (changed with `Pipeline::set_param`, see below). The description of each parameter says which.

### Edges

`{ "from": "<input or node>", "to": "<node or output>" }`, with the optional `"port"`. A source may have any number of outgoing edges (fan-out; all readers share one buffer, no copy). The graph must be acyclic. The output of every node must be connected to a node or an output. An input without an outgoing edge is only a warning.

An input may be connected directly to an output.

**Ports.** A node has one input port, `a`, or two, `a` and `b` (the kinds `add`, `mul`, `make_complex` and `stereo_pll`; the first input in the description of the block is `a`, the second `b`). An edge feeds the port named in its `port` field; without the field it feeds `a`:

```json
{ "from": "quad", "to": "pll", "port": "a" },
{ "from": "pilot", "to": "pll", "port": "b" }
```

Every input port of a node and every output has exactly one incoming edge. Both ports may read the same source. An output has no ports (`port` on an edge to an output is `unknown_port`), and a node of a one-input kind accepts `a` only. The short form `"to": "pll.b"` is not supported: names contain no dots, and one way is enough (D380). `to_json` writes `port` exactly when the description had it, so a flow that never names a port round-trips unchanged and an explicit `"a"` stays (D381).

**Rules for a node with two inputs.**

- *Type.* Both ports take the same type; a complex stream into `make_complex` or `stereo_pll`, or an `f32` stream beside a `C32` stream, is `type_mismatch` at the edge of the port that does not fit (port `b`: the type of port `a`).
- *Rate.* Both streams have the same exact sample rate, otherwise `rate_mismatch` at `edges[i].to` of the port-`b` edge; the message names both rates.
- *Clock.* Both streams derive from the same pipeline input. Two inputs of the same nominal rate come from different devices and drift, so the check does not look at the rates: `clock_domain` at the port-`b` edge ("resample one with a tunable `resample` node driven by a drift controller, or use one input"). Every node of the report has its `clock_domain`, the name of the input its data derives from. Packet ports carry no clock domain and are exempt (below).
- *Delay.* The two paths can have different group delays. The build adds up the delays from the input along each path (exact rationals) and compares them in samples of the node's input rate. If they are equal nothing happens. If they differ by a whole number `k`, the build inserts the automatic node `<node>__align` (kind `delay`, `samples` = `k`, `auto: true` in the report) in front of the port with the smaller delay. If the node says `"align": "none"`, nothing is inserted and the warning `misaligned_inputs` at `nodes.<name>` gives the difference. If the difference is not a whole number the description is wrong: `misaligned_inputs` at `nodes.<name>` with both delays in the message; change a filter so that the delays fit (a fractional delay is in the backlog). A difference of more than 65 536 samples is the same error. `<node>__align` is reserved like the other automatic names.

**Packet edges** (M37). Packet ports connect only to packet ports: a packet source (a packet input or a decoder such as `frame_decode`) into a stream port, or a stream into a packet port, is `type_mismatch` at `edges[i].to`; a packet source into a `"stream"` or `"frames"` output, or a stream into a `"packets"` output, is `type_mismatch` at `outputs.<name>.type`. A packet source may fan out like a stream: every reader (an encoder, port `b` of a clocked encoder, an output) gets every packet. A packet input may go directly to a packet output (the packets come out unchanged, index and flags included).

**Clocked nodes.** A kind whose second input port takes packets (`Clocked<S, Packets>` in [blocks.md](blocks.md#packet-ports); none of the standard kinds has one yet, the first is `rds_encode` of M42) reads its clock stream at port `a` and the packets at port `b`. The rules for a node with two inputs apply to port `a` only: port `b` has no type pairing with `a`, no rate, no clock domain and no group delay, so the build neither checks nor aligns it, and no `<node>__align` is inserted for it. The runner calls the node when port `a` holds samples (with any number of queued packets, also none) and advances port `b` by the packets the block reports as consumed (`Node::consumed_packets`).

### Outputs

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"stream"`, `"frames"` or `"packets"` | `"stream"` takes the output of a block, `"frames"` the output of a frame block (`spectrum`), `"packets"` the output of a decoder (`frame_decode`) or a packet input. The type must match what the source produces. |
| `format` | name or object, optional | Stream outputs only: convert to raw bytes in this format (an automatic `convert_out` node). Only complex streams can be converted. |
| `capacity` | integer, optional | Stream outputs only: size of the output queue in samples (bytes for a raw output), 1 to 2^28. Default `4 * max_chunk`. |
| `queue` | integer, optional | Frames and packets outputs only: length of the queue in frames, 1 to 65 536, default 4; or in packets, 1 to 65 536, default 64 (`DEFAULT_PACKET_QUEUE`). A packets output has no `format` and no `capacity` (`invalid_value`). |
| `drop` | `"error"`, `"drop_oldest"`, `"drop_newest"`, optional | What happens when the output is full. Default `"error"`. |
| `underrun` | `"zeros"`, `"short"`, optional | Pull mode: what `pull` does when the source is exhausted. Default `"zeros"` for raw outputs, `"short"` otherwise. |
| `description` | string, optional | |

Drop policies:

| Policy | Behaviour |
| --- | --- |
| `error` | `push` fails with `OutputFull` **before it consumes anything** when the worst-case result of the push (`ceil(len * R)` elements, with `R` the rate ratio from the input to the output, one more for rounding; capped at the capacity of the output) does not fit. Nothing is lost. If a block still produces more than estimated, the blocked nodes keep their data until the reader makes room (back-pressure); reading from the output continues the processing. |
| `drop_oldest` | The oldest data of the output is discarded to make room. |
| `drop_newest` | Data that does not fit is discarded. |

`Pipeline::dropped(output)` counts the discarded elements (frames for a frames output, packets for a packets output). For a packets output the policies count packets, and `R` of the `error` policy is the product of the timing ratios of the nodes from the packet input to the output: the most samples an encoder makes of one packet (`frame_encode`: `8 * (preamble_bytes + 260)`, 2112 with the defaults) times the most packets a decoder finds per sample (`frame_decode`: 1/56), about 38 packets per pushed packet. A packet input has only a nominal rate, so the rate ratio would not bound a push (D513). The bound is large because a decoder may find a packet in every 56 samples of noise; a flow that must not lose packets gives its output a large `queue` and reads after every push (`frame_loop.json`: 4096).

### Rust types

`PipelineSpec`, `Settings`, `InputSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `GroupSpec`, `BackendSpec`, `Precision`, `DropPolicy`, `UnderrunPolicy` (and the helper types `RateHz`, `FormatSpec`, `FormatName`, `CustomFormat`, `InputType`, `OutputType`) are plain `serde` structs with `#[serde(deny_unknown_fields)]`.

```rust
let spec = PipelineSpec::from_json(text)?;   // Err(Error::Json) with the position of a problem
let canonical = spec.to_json();               // pretty, keys in field order, maps sorted
```

`to_json(from_json(x))` parses back to an equal spec, and the files in `examples/flows` are canonical (the bytes are identical). Duplicate keys inside one JSON object are not detected (the last one wins), because `serde_json` does not report them.

## Registry and factories

`Registry::standard()` registers every block of `sdrk-blocks`. Each kind has a `BlockFactory`: it parses and checks the parameters for the input type, rate and precision (`check`), names the kernels the block calls (`kernels`), and builds the type-erased `Node` (`build`). `Checked` carries the figures of a checked node: output rate, memory, cost, output kind, number of taps, group delay and the smallest input count that produces an output.

`BlockFactory::inputs()` lists the input ports of a kind (`PortInfo { name, accepts }`, `a` first), taken from the description of the block. The runner works on type-erased nodes: `NodeInput`/`NodeOutput` are typed, borrowed views, `StreamNode<B: Block>` (stream input and output), `PairNode<B: Block<In = Two<X, Y>>>` (two stream inputs, passed as `NodeInput::Two { a, b }` with one `NodeElem` per port) and `FrameNode<B: Block<Out = Frame>>` wrap the generic blocks. A variant mismatch inside `process` is an `Internal` error that cannot happen after a successful build.

## Validation

`Registry::validate(&spec, &opts)` and `Registry::validate_json(text, &opts)` return a `Report` with **all** problems at once. Nothing is built; the figures are estimates from the configurations.

Validation has two stages:

1. **Structure.** For JSON text: syntax (with line and column), unknown fields, JSON types, enumerations, required fields, the schema version. Then, on the typed description: names, descriptions, kinds, parameter names, types and absolute ranges, output attributes.
2. **Graph.** Endpoints, the ports of the edges, one incoming edge per input port and output, dangling outputs, cycles (Kahn's algorithm; one node of each cycle is reported), the port types at every edge, then in topological order the rates with exact `Rate` arithmetic, for a node with two inputs the checks of rate, clock and delay with the automatic alignment, the rate-dependent checks of every block, and the limits.

Stage 2 runs only if stage 1 found no error, so a single typo does not cause a flood of follow-up diagnostics.

`Report::to_json()` is the machine-readable format for agents:

```json
{
  "errors": [
    { "path": "nodes.tune.params.frequency_hz", "code": "rate_unsupported", "message": "parameter frequency_hz out of range: ..." }
  ],
  "warnings": [],
  "memory_bytes": 0,
  "cost_per_input_sample": 0.0,
  "nodes": [],
  "outputs": []
}
```

`path` uses dots and the JSON names: `nodes.tune.params.frequency_hz`, `edges[2].to`, `inputs.rf.format`. The node report lists name, kind, input and output rate (an integer, or `"num/den"`), the group delay in input samples (`"num/den"`), whether the node is automatic and its `clock_domain`. `outputs` has one entry per output with its `group_delay` (in samples of the output, `"num/den"`) and `max_buffered` (see [timing.md](timing.md#latency)). `group_edges` (only present with groups) lists the edges between nodes of different groups.

### Diagnostic codes

| Code | Stage | Meaning | Path |
| --- | --- | --- | --- |
| `syntax` | 1 | Not valid JSON (message carries line and column), or the document does not fit the model. | `""` |
| `schema_version` | 1 | `sdrk` is missing or not 1. | `sdrk` |
| `unknown_field` | 1 | A key that the model does not have. | the key |
| `invalid_name` | 1 | A name that does not match `^[a-z][a-z0-9_]{0,63}$`. | `nodes.<name>` etc. |
| `duplicate_name` | 1 | A name used twice among inputs, nodes and outputs, or one that collides with an automatic node. | the second use |
| `unknown_kind` | 1 | A node kind that is not registered. | `nodes.<name>.kind` |
| `unknown_param` | 1 | A parameter the block does not have. | `nodes.<name>.params.<param>` |
| `wrong_type` | 1 | A JSON value of the wrong type. | the value |
| `out_of_range` | 1 | A number outside the limits of the parameter or field. | the value |
| `missing_param` | 1 | A required field or parameter is absent. | the missing key |
| `invalid_value` | 1 | The right type but not a possible value: unknown option or format name, non-integer for an integer, a format that cannot stream, inconsistent combinations. | the value |
| `unknown_endpoint` | 2 | An edge names something that does not exist or cannot be used on that side. | `edges[i].from` / `edges[i].to` |
| `multiple_inputs` | 2 | An input port of a node, or an output, with more than one incoming edge. | `edges[i].to` of the extra edge (port `a`, outputs), `edges[i].port` (port `b`) |
| `missing_input` | 2 | An input port of a node, or an output, without an incoming edge. | `nodes.<name>` (port `a`) / `nodes.<name>.ports.b` / `outputs.<name>` |
| `dangling_output` | 2 | The output of a node goes nowhere. (Warning: an input nobody reads.) | `nodes.<name>` / `inputs.<name>` |
| `cycle` | 2 | A node on a cycle. | `nodes.<name>` |
| `type_mismatch` | 2 | Port types that do not fit: a block fed with the wrong sample type, frames into a node, a stream output on frames or the other way round, an input type that does not fit the precision. | `edges[i].to`, `outputs.<name>.type`, `inputs.<name>.type` |
| `rate_unsupported` | 2 | The block rejects its configuration for the sample rate at its input (frequency beyond Nyquist, a filter that does not fit, ...). | `nodes.<name>.params.<param>` |
| `limit_exceeded` | 2 | A resource limit is exceeded. | `nodes.<name>` (taps), `nodes` (the rest) |
| `unknown_group` | 1 | A node names a group that `groups` does not declare. | `nodes.<name>.group` |
| `backend_unavailable` | 2 | A required backend does not implement the kernels of the node (an error for `require`, a warning for `prefer`). | `nodes.<name>.backend.require` / `.prefer` |
| `unknown_port` | 2 | An edge names an input port that its destination does not have: `port` on an edge to an output, or `b` on a node with one input. | `edges[i].port` |
| `rate_mismatch` | 2 | The two inputs of a node have different sample rates. | `edges[i].to` of the port-`b` edge |
| `clock_domain` | 2 | The two inputs of a node derive from different pipeline inputs. | `edges[i].to` of the port-`b` edge |
| `misaligned_inputs` | 2 | The group delays of the two paths into a node differ by a fraction of a sample or by more than 65 536 samples (an error); they differ and `align` is `none` (a warning). | `nodes.<name>` |

The invalid-flow corpus in `crates/pipeline/sdrk-pipeline/tests/invalid` has one file per code, with the expected diagnostic in a sibling `.expected.json`.

## Build

`Registry::build(&spec, &opts)` validates first and returns the report as the error if there are errors.

- **Automatic nodes.** An input with a raw `format` gets a node `<input>__convert` (kind `convert`; the real variant for a format with a real layout), and every edge from that input reads the converted stream. A stream output with a `format` gets `<output>__convert_out` (kind `convert_out`). A node with two inputs whose paths differ in group delay gets `<node>__align` (kind `delay`) in front of the faster port. Automatic nodes appear in the report (`auto: true`) and in the statistics, but not in the readout of `config()`, which reproduces the user's description.
- **Order.** Topological order; ties are broken by the first appearance of a node in `edges`. The automatic input converters come first, the output converters last.
- **Buffers.** One linear buffer per node output and per input, shared by all readers: each reader has its own position (zero-copy fan-out). The capacity of the buffer after node `k` is `ceil(max_chunk * R_k) + min_input + 64` elements, where `R_k` is the rate of the buffer divided by the rate of its input (exact; times the bytes per sample for a raw buffer) and `min_input` is the largest `min_input` of the readers. The two ports of a node run at the same rate and come from the same input, so both buffers use the same `R_k`. A writer compacts the buffer (moves unread data to the front) when the free tail is smaller than what the node could write, so every block sees contiguous input. Every output has its own queue behind the buffer, which is where the drop policies apply.
- **Packet buffers.** A buffer of packets holds `ceil(max_chunk * R) + 4` packets, without `min_input` and the slack of 64: for the buffer of a packet input `R` = 1 (`max_chunk` packets per push), behind a decoder `R` = `max_per_input` of its timing (`frame_decode`: 1/56). For this the stream of a packet input gets the nominal rate `rate_hz` (1 Hz without it), the stream behind an encoder the rate `rate_hz` and the stream behind a decoder the rate of its input times `max_per_input` (D513). An encoder continues a frame that does not fit its output buffer in the next call, and a decoder stops when its output buffer is full, so these sizes never lose data. `Report::memory_bytes` counts `size_of::<Packet>()` (272 bytes) per packet slot of the buffers and queues.
- **Memory.** Buffers, queues and block storage (`storage_bytes`). `Report::memory_bytes` of `validate` is an estimate; after `build` the report of the pipeline has the exact figure.
- **Cost** (an estimate of multiply-accumulates; see below).

### Limits

`BuildOptions { limits }`:

| Limit | Default | Violation |
| --- | --- | --- |
| `max_nodes` | 256 | `limit_exceeded` at `nodes` (automatic nodes count) |
| `max_memory_bytes` | 256 MiB | `limit_exceeded` at `nodes` |
| `max_cost_per_input_sample` | infinite | `limit_exceeded` at `nodes` |
| `max_taps_per_node` | 16 383 | `limit_exceeded` at `nodes.<name>` |

### Stereo audio

A stereo audio stream is a stream of the complex sample type of the precision: **`re` is the left channel, `im` the right channel** (`C32` for `f32`, `CQ15` for Q15). No port type `stereo` exists, because none is needed:

- `C32` and `CQ15` are `#[repr(C)]` with `re` first, so a stream of them is exactly the interleaved two-channel layout (L, R, L, R, ...) that sound cards and WAV files use; a reader copies the bytes without reordering.
- Every block that works on complex streams works on stereo audio unchanged and treats both channels alike: the `resample` (also tunable, for the clock drift of a sound card), `scale`, `fir`, `biquad`. `ddc`, `squelch`, `fm_demod` and `spectrum` accept the type too but make no sense on audio; the pipeline does not prevent such a flow.
- `sdrk-io` writes `CF32` and `CS16` as two-channel WAV files with I in the left channel, which is the same convention (the `fm-receiver` example writes a stereo file when the output `audio` is a `C32` stream).
- A port type would need changes in the pipeline model, the schema, the wire format, the WASM bindings and the I/O crate, for no new capability (decision D122 in [decisions-log.md](decisions-log.md)).

The kind `fm_stereo_demod` produces such a stream from a complex FM signal at 120 kHz or more (`rate_unsupported` below); see [blocks.md](blocks.md). On the transmit side (M36) `preemph_limit` takes a `c32` stereo stream (one gain for both channels) and `fm_stereo_mod` turns it into the real multiplex signal for `fm_mod`; both are `f32` only (a `cq15` stereo stream is a `type_mismatch`).

### Symbols and soft bits

`symbol_sync` (M38) turns a matched-filtered `c32` or `f32` stream into one sample per symbol: its output stream has the nominal rate `symbol_rate_hz` (rounded to 1/1000 Hz) and its timing is `Variable` (the loop follows the clock of the signal within `max_deviation`), so the buffer behind it is sized from the nominal rate plus `min_input` and the slack of 64, and the block stops when that buffer is full and continues with the rest of its input in the next step. `slice` turns symbols into `f32` soft bits (above 0 is bit 1, [blocks.md](blocks.md#soft-bits)) at the symbol rate times the bits per symbol; its `scale` is a run-time parameter. A soft-bit stream feeds `frame_decode`. Both kinds take only `c32` and `f32` streams (no Q15 variants, D529); `cq15` and `q15` are a `type_mismatch`.

PSK (M40): `psk_mod` turns `f32` soft bits into a root-raised-cosine shaped `c32` stream at `samples_per_symbol` samples per symbol (constant ratio `sps / bits per symbol`, BPSK or QPSK, optionally differential); `psk_demod` turns a `c32` stream with an integer number of samples per symbol (within 1 %; otherwise `rate_unsupported` at `params.symbol_rate_hz`, with the hint to put a `resample` before it) into soft bits through the matched filter, `symbol_sync` and the Costas loop (or differential detection). Its timing is `Variable` like `symbol_sync`; soft bits that do not fit into the buffer behind it wait in the block. `costas` alone runs the carrier loop on one `c32` sample per symbol (for a receiver built from `fir`, `symbol_sync`, `costas` and `slice`). The three kinds take only `c32` (`psk_mod` only `f32`) and have only build parameters; `order` 3 is `invalid_value`. For coherent QPSK the 90 degree ambiguity is not resolved; `frame_decode` with `polarity` `both` resolves 180 degrees (BPSK).

### Cost formulas

The cost is an estimate, in multiply-accumulates (a complex tap counts like the formulas below say), not a measurement. Each factory gives the cost **per input sample of the node**:

| Kind | Cost |
| --- | --- |
| `convert`, `convert_out` | 2 |
| `scale` | 1 |
| `dc_block` (complex or real input), `squelch` | 3 |
| `agc` | 6 |
| `fir` | `taps` |
| `ddc` | `4 + 4 * taps / decimation` |
| `duc` | `4 * interpolation + 4 * taps` |
| `fm_demod` | `10 + taps / audio_decimation` (`taps` of the audio filter, 0 without it) |
| `fm_stereo_demod` | `25 + (2 * taps + 12) / audio_decimation` (`taps` of the audio filter; the pilot band-pass, the loop, the complex filter, the matrix and the de-emphasis) |
| `am_demod` | `4 + taps / audio_decimation` (`taps` of the audio filter, 0 without it) |
| `ssb_demod` | `8 + 4 * taps / audio_decimation` (two NCOs and the complex filter, as `ddc`) |
| `halfband_decimate`, `real_to_iq` | `(N + 3) / 4` (`N = 4 k + 3` taps; the `(N + 3) / 2` non-zero taps once per two input samples) |
| `convert` (real format) | 1 (2 for a complex format) |
| `biquad` | `5 * sections` (real input) or `10 * sections` (complex input); `sections = order / 2` for low-pass and high-pass, 1 for band-pass and notch |
| `fm_mod` | 6 |
| `am_mod` | 2 (real input only, `f32` or `Q15`) |
| `ssb_mod` | `8 + 4 * taps` (two NCOs and the complex filter at the audio rate; real input only) |
| `quad_demod` | 12 |
| `deemphasis` | 3 |
| `rotate` | 6 |
| `decimate` | `taps / factor` (`taps` of the designed low-pass) |
| `interpolate` | `taps` (`taps` of the designed low-pass at the output rate) |
| `mag` | 4 |
| `delay` | 1 |
| `preemph_limit` | 6 for a real stream (one pole, the envelope and the gain); 12 for a `c32` stereo stream (two poles, one envelope and one gain for both channels); `cq15` is a `type_mismatch` |
| `fm_stereo_mod` | 10 (sum and difference, two interpolated table lookups, three products; `c32` input only, `rate_unsupported` below 120 kHz) |
| `to_complex`, `real_part`, `imag_part` | 1 |
| `nco` | 6 (the table lookups, the interpolation and the phase step) |
| `stereo_matrix` | 2 |
| `noise_blank` | 8 (the power, the impulse test, the average and the delay line; complex input only, `C32` or `CQ15`, a real stream is a `type_mismatch`) |
| `auto_notch` | `2 * taps + 4` (the prediction and the update of the weights; `f32` input only, Q15 is a `type_mismatch`) |
| `resample`, rational | `taps / M`, `M` the denominator of `rate_out / rate_in` |
| `resample`, tunable | `K * rate_out / rate_in + 4`, `K` taps per phase |
| `spectrum` | `(2.5 * N * log2(N) + 3 * N) / interval_samples`, `N = fft_size` |
| `frame_encode` | 1 per output sample (its `rate_in` is the `rate_hz` of the packet input) |
| `frame_decode` | 1 |
| `symbol_sync` | `2 + 24 / sps` (two cubic interpolations per symbol, `sps = rate_in / symbol_rate_hz`; `c32` or `f32` input, `rate_unsupported` below 2 samples per symbol) |
| `slice` | 1 (one product per soft bit; `real` takes `f32`, `bpsk` and `qpsk` take `c32`, the other combinations are `invalid_value` at `params.constellation`) |
| `costas` | 30 (two table interpolations, the derotation, the detector, the loop and the lock detector per symbol; `c32` input only, D615) |
| `psk_mod` | `4 * span` (`f32` soft bits only) |
| `psk_demod` | `4 * span + 40 / sps` (`sps = rate_in / symbol_rate_hz`, within 1 % of an integer from 2 to 1024 and at least 2, else `rate_unsupported`; `c32` input only) |

The cost of the pipeline is `sum(node cost * samples per second at the node) / sum(input samples per second)`: MACs per pipeline input sample, with a node behind a decimator counted at the lower rate. A packet input counts with its `rate_hz` (the samples its encoder makes); one without `rate_hz` does not count.

## Runner (push mode)

```rust
let mut pipeline = registry.build(&spec, &BuildOptions::default())?;
pipeline.push("rf", InputData::Raw(&bytes))?;           // up to max_chunk samples
let n = pipeline.read_stream("audio", &mut audio)?;     // T must match the output type
while let Some(meta) = pipeline.read_frame("spectrum", &mut frame)? { /* ... */ }
pipeline.set_param("tune", "frequency_hz", serde_json::json!(610_000.0))?;
```

- **`push`.** Feeds up to `max_chunk` samples to one input and runs the graph until no node can make progress. All of `data` is consumed, or an error is returned before anything is consumed. When the buffer of the input is empty, the first readers process the caller's slice directly; only what they did not take (history-limited `min_input`, a split packed sample) is copied into the input buffer, once.
- **Nodes with two inputs.** The runner calls the node when **both** buffers hold at least `min_input` samples and passes `n = min(available_a, available_b, room)` samples of each (lockstep); both read positions advance by the consumed count. Such a node never works on the caller's slice directly (a node that reads the pipeline input at port `a` or `b` takes its data from the input buffer). Each `push` gives both paths the same number of samples after their decimations, so no buffer grows without bound; a debug build asserts at every step that the fills of the two buffers differ by less than the capacity of a buffer.
- **`push_packets(input, &[Packet])`.** `push` with `InputData::Packets` for a packet input: all packets or none (the same checks as `push`, `ChunkTooLarge` above `max_chunk` packets). A pure encoder produces the samples of the packets it has; with no packet queued it produces nothing (no idle signal in push mode, D515). A packet whose payload the encoder cannot frame (`frame_encode`: more than 253 bytes) fails with `Error::Block` when the encoder reaches it. `push_with_meta` on a packet input is `WrongType`: packets carry no block metadata.
- **Decoders.** The runner moves the packet `index` of a decoder from the count of the block (input samples since its construction or reset) into the index domain of the node's input: `index` is the input sample index of the first sync word bit, counted like the samples of that stream (behind an encoder: its output samples since the build).
- **`read_stream::<T>`, `read_raw`, `read_frame`, `read_packets`, `available`.** Read from the output queues. `read_stream` is for `C32`, `CQ15`, `f32` and `Q15` outputs, `read_raw` for outputs in a raw format, `read_frame` for frame outputs (the buffer must have the frame length), `read_packets(output, &mut [Packet]) -> usize` for packets outputs (oldest first). `available` counts samples (bytes for a raw output), frames or packets. `output_port` of a packets output is `PortKind::Stream(PortType::Packets)` (D512). Packet files in the JSON Lines format are read and written by `sdrk_io::packets` ([io.md](io.md#packet-files-json-lines)); the WebAssembly binding has `pushPackets` and `readPackets` ([wasm.md](wasm.md#packets)).
- **`set_param(node, name, value)`.** Changes a runtime parameter immediately: it applies from the next sample the node processes. A build parameter is `ReadOnlyParameter`. The call does not allocate when it succeeds.
- **`config()`.** The description of the pipeline as built, with the current values of the runtime parameters that were changed. For an unchanged pipeline `to_json(config())` equals `to_json(from_json(input))`.
- **`stats()`.** Per node (automatic ones included): `calls`, `samples_in`, `samples_out`, `frames`.
- **`reset()`.** Resets all blocks, empties all buffers and queues; statistics are kept.
- **`report()`.** The validation report with the exact delays of the built blocks.

### Metadata and events

```rust
pipeline.push_with_meta("rf", InputData::Raw(&bytes), &meta)?;  // meta: BlockMeta { start, flags, anchor, .. }
while let Some(event) = pipeline.poll_event() { /* Event { output, index, kind } */ }
let name = pipeline.output_name(event.output);
```

Every stream has its own sample index (`u64`, see [timing.md](timing.md)). `push_with_meta` is `push` with the metadata of the block:

- The **first** block of an input fixes the origin of its index (`meta.start`); with plain `push` it is 0. Joining a stream at sample 10^9 gives the same results as starting at 0, apart from the indices.
- A start **above** the expected index is a gap: everything that depends on the input is reset and the origins move by the mapped gap, so the result equals two separately reset runs. A `Gap { missing }` event is reported on every output reachable from the input.
- A start **below** the expected index is `InvalidState`: time does not go backwards. The expected index is `input_index(input)`.
- The flags `OVERFLOW`, `UNDERRUN`, `RETUNED`, `LATE`, `BURST_START`, `BURST_END` and the anchor become events at the mapped positions of **every** output reachable from the input (one event per output). `BURST_END` is at the last sample of the block, the other flags at its first sample.
- Raw data must hold whole packing groups (`LengthMismatch` otherwise).

`Event { output: OutputId, index: u64, kind: EventKind }` with `EventKind` = `Gap { missing }`, `Overflow`, `Underrun`, `Anchor(TimeAnchor)`, `Retuned`, `BurstStart { tx_time_ns }`, `BurstEnd`, `Late`, `ParamApplied { node, param, value, late }`, `EventsLost { count }` (`#[non_exhaustive]`). For a frames output the index is the index in the input of the frame node; `FrameMeta.first_sample` of a frame is an absolute index of that input as well.

The queue holds `settings.event_queue` events (default 256) and is allocated at build. When it is full the newest events are dropped and counted; once the queue is empty `poll_event` returns `EventsLost { count }` with the output and index of the first lost event. `poll_event` and `push_with_meta` never allocate (tested).

Further accessors: `input_index(input)`, `read_index(output)` (index of the next sample `read_stream` returns), `time_of(output, index)` (absolute time from the latest anchor of the output, `None` without one), `node_id`, `output_id`, `node_name`, `output_name`, `latency(output)`, `input_rate(input)` and `output_rate(output)` (the exact `Rate` of a stream; for a frames output the rate at the input of the frame node), `input_port(input)` (`(PortType, max_len)`: the kind of data an input takes and the most elements one `push` accepts, bytes for a raw input) and `output_port(output)` (`PortKind::Stream(PortType)` or `PortKind::Frames { len }`). The last two let a binding size its buffers and pick the typed array without parsing the description ([wasm.md](wasm.md)).

### Timed parameters

```rust
pipeline.set_param_at("tune", "frequency_hz", json!(610_000.0),
    ParamTime::InputIndex { input: "rf".into(), index: 1_234_567 })?;
```

`ParamTime` is `Immediate`, `InputIndex { input, index }` (a sample of a pipeline input) or `UnixNs(i64)` (needs an anchor on the input of the node: `n = anchor.index + ceil((t - anchor.unix_ns) * rate / 1e9)`; without one `InvalidState`). The target is mapped to the index domain of the input of the node with the mapping rule. The runner ends the next `process` call of the node exactly at that sample, sets the parameter and continues, so the change is visible at the first output sample computed after it and never changes samples that were computed before. The acknowledgement is a `ParamApplied` event on every output that depends on the node, at the mapped index of the effective sample.

Up to `settings.pending_params` changes (default 16) wait per node, ordered by their target (`PendingFull` when the list is full). A target that has already passed is applied at once with `late: true`. The calls validate the parameter like `set_param` (unknown, build-time, wrong type, out of range).

### Pull mode (transmit)

```rust
struct Audio { /* ... */ }
impl PullSource for Audio {
    fn provide(&mut self, input: &str, max_samples: usize) -> Result<InputData<'_>, PullError> { /* ... */ }
}
let (n, meta) = pipeline.pull("rf", &mut bytes, &mut audio)?;       // raw output
let (n, meta) = pipeline.pull_samples::<f32>("pcm", &mut pcm, &mut audio)?;
```

The output asks, the source supplies. `pull` produces exactly `out.len()` bytes (whole samples only; `pull_samples` samples) unless the source is exhausted: while the queue holds fewer elements than requested it asks `provide(input, ceil(missing / R) + min_input)`, at most `max_chunk`, pushes the result and runs. Surplus production stays in the output queue for the next call, so the bytes do not depend on the sizes of the pulls; they equal a push-mode run of the same input. An empty result from `provide` means that the source is exhausted. The output setting `underrun` then decides: `"zeros"` (default for raw outputs) fills the rest of the request with silence, flags the block `UNDERRUN` and reports an `Underrun` event at the first silent sample; `"short"` returns fewer samples. Silence is a zero sample **in the format of the output** (`0x80` bytes for `cu8`, found by converting a zero sample with the `convert_out` node of the output). The zero samples are not part of the index space of the output.

The returned `BlockMeta.start` is the output index of the first sample. A block never spans a burst boundary:

- `begin_burst(output, tx_time_ns)`: the next sample the output produces is the first sample of a burst. The block that contains it carries `BURST_START` and `tx_time_ns`; samples that were already buffered come first, in a block of their own. A `BurstStart` event is reported at the same index.
- `end_burst(output)`: feeds `ceil(D_in)` zero samples to the input (`D_in` is the group delay of the path in *input* samples), so that the filter tails come out. The burst ends `ceil(lead_time)` samples after the sample that the last real input sample leads to (`lead_time(output)`), the block that ends there carries `BURST_END` and a `BurstEnd` event is reported. Whatever the zero samples produced beyond that is discarded and the nodes of the path are reset when the tail has been pulled. Pull the data of the burst before calling `end_burst`: the zero samples need room in the output (`OutputFull` otherwise).
- Between `end_burst` (after the tail was pulled) and the next `begin_burst` nothing is produced and the source is not asked. Without any `begin_burst` the output runs continuously.
- The output index continues across bursts and never goes back. (The samples that the flush produced beyond the end of the burst are skipped in the index space, fewer than the interpolation of the path.)

`device_status(output, &meta)` forwards what `TxStream::poll_status` returned (underrun, late, burst done) as events of the output at index `meta.start`. `lead_time(output)` is the cumulative group delay of the path in output samples.

A node with two inputs changes nothing in pull mode: both of its paths come from the same input, so `provide` is called once per round, with the input of the path. `end_burst` flushes and resets every node the output depends on, the branches that join at port `b` included.

`pull` does not allocate; `end_burst` does (the zero samples).

**Packet inputs in pull mode** (M37). `provide` may return `InputData::Packets(&[Packet])` for a packet input (at most `max_samples` packets). An empty slice means "nothing to send now", not "exhausted": the encoder at the start of the path (`frame_encode`) is called with no packets and room for the samples the output still needs, mapped to its own rate, and fills them with its idle signal (`idle`: `zero` or `preamble`). When that share does not move the output (a block on the path needs more input before it produces, such as a QPSK `psk_mod` that takes two bits per symbol), the encoder is called again, up to 64 times per step (D620). Packets that wait in the input buffer are encoded first; an idle call never cuts a frame. So a packet input never underruns, and `underrun` does not apply to it; a pull is always complete. A packet input ends only with `end_burst` of the output: it feeds no zero samples but lets the encoder send `ceil(D_in)` idle samples (`D_in` counts samples of the encoder output, whose rate is the `rate_hz` of the input), and the burst ends after the samples that the encoder had produced when `end_burst` was called, plus `ceil(lead_time)` (D516).

### Clock drift

```rust
let config = DriftConfig::new(44_100.0, 4_410.0);
let mut drift = PipelineDrift::new(&pipeline, "audio_out", "to_cd", config)?;
let ppm = drift.update(&mut pipeline, now_s, extra_fill)?;   // now_s: monotonic seconds
```

`sdrk_pipeline::drift::PipelineDrift` keeps the queue of a stream output at a target fill by steering the `ppm` parameter of a tunable `resample` node with the `DriftController` of `sdrk-blocks` (the control law and the configuration are in [blocks.md](blocks.md), section "Clock drift"). `new(pipeline, output, node, config)` requires a stream output and a `resample` node with `tunable = true` on the path to it; otherwise it returns `Error::Build` with one `invalid_value` diagnostic (an unknown output or node is `UnknownOutput` or `UnknownNode`, an invalid configuration is `Error::Block`). `update(pipeline, now_s, extra_fill)` reads `available(output) + extra_fill`, runs the controller and calls `set_param(node, "ppm", ..)`; it returns the value set and does not allocate (tested with a counting allocator). `controller()` and `controller_mut()` give access to the controller (for example `set_initial_ppm`). Call `update` immediately before every `push` to the input and immediately before every read of the output (in an audio callback: update first, then take the block): each call then sees the level that held since the previous call, and the time-weighted smoothing of the controller averages the queue over time. Calling it only after each read samples the queue at one phase of the output clock; when the input and output blocks have about the same period (480 in, 441 out) that reading jumps by a whole block whenever the two clocks slip by one period, and the controller answers each jump with a spike of several hundred ppm. With the calling contract the `resample_audio` test at +50 ppm settles within 1.74 ppm of -50 ppm (D314). The application decides where the buffer is and when to call `update`; the pipeline never inserts the controller by itself.

### Latency

`latency(output)` returns `Latency { group_delay, max_buffered }`, also reported per output in `Report.outputs`. [timing.md](timing.md#latency) explains how both add up.

## Groups and backend preferences

Groups and preferences are part of the model only; they have **no runtime effect yet**. They are the single extension point for threads, MCU cores, GPUs and network splits (see [architecture.md](architecture.md)).

```json
"groups": { "front": { "description": "DDC on core 0" }, "audio": {} },
"nodes":  { "tune": { "kind": "ddc", "group": "front", "backend": { "require": "simd" } } }
```

- A node names a group (`UnknownGroup` if it is not declared); nodes without one are in the implicit group `"main"`, which may also be named without declaring it. A group name follows the name rule.
- `Report.group_edges` lists every edge between nodes of different groups: these edges are the interfaces. Inputs and outputs have no group: an input belongs to every group that reads it, an output to the group of the node that feeds it.
- `backend` is `{"prefer": name}` or `{"require": name}`. The factory names the kernels the block calls (`BlockFactory::kernels`); `require` fails with `backend_unavailable` if `sdrk::backend_for(kernel, type)` returns another name for one of them, `prefer` only warns. A kernel is looked up for the complex and the real type of the precision, whichever the facade lists; fused kernels (`chained`) are not judged, their parts are. The backends are `simd` (the accelerated kernels, see [backends.md](backends.md)) and `fallback`; a build with `--cfg sdrk_fallback_only` reports `fallback` for every kernel.
- An edge that carries packets cannot cross groups yet (`invalid_value` at `edges.<from>` from `split_by_groups`): the group input would need the rate of the stream an encoder makes from it, which the edge does not carry (D517).
- `split_by_groups(&spec, &registry)` returns one description per group, sorted by group name. A cut edge `a -> b` becomes the output `edge__a__b` (a stream of the type of `a`) in the group of `a` and the input `edge__a__b` (the type and the exact rate) in the group of `b`. (The name rule does not allow a leading underscore, so the prefix is `edge__`.) Pipeline inputs are copied into every part that reads them.

## YAML

With the feature `yaml` (default; the WASM binding disables it) `PipelineSpec::from_yaml(text)` reads the same model from YAML, using `serde-saphyr` (version 2.0.0). Unknown fields are errors as in JSON, comments are allowed and errors carry line and column. `Registry::validate_yaml` and `Registry::build_yaml` give the same diagnostics (same codes, same paths) as the JSON functions; JSON text is also valid YAML. There is no YAML writer: JSON is canonical. Every example flow has a `.yaml` twin with comments.

## Wire format

[wire-format.md](wire-format.md) specifies the versioned messages (`wire` module) that carry samples, frames, events, stream descriptions and parameter changes between the parts of a split pipeline. Version 1 carries no packets: `Message::from_input` makes an empty raw block of `InputData::Packets` (D517).

## Errors

`Error` is `#[non_exhaustive]`, implements `SdrkError` and maps to the shared `ErrorKind`:

| Variant | Kind | Meaning |
| --- | --- | --- |
| `UnknownInput`, `UnknownOutput`, `UnknownNode` | `InvalidArgument` | The name is not declared. |
| `WrongType` | `InvalidArgument` | The data or the sample type does not match the input or output. |
| `ChunkTooLarge` | `LengthMismatch` | More than `max_chunk` samples in one push. |
| `OutputFull` | `CapacityExceeded` | An `error` output cannot take the worst case; nothing was consumed. |
| `LengthMismatch` | `LengthMismatch` | A frame buffer of the wrong length. |
| `InvalidParameter` | `InvalidArgument` | A runtime parameter value of the wrong JSON type or not a possible value. |
| `Block { node, source }` | kind of `source` | A block failed (`UnknownParameter`, `ReadOnlyParameter`, `OutOfRange`, ...). |
| `InvalidState` | `InvalidState` | Metadata that goes back in time, a timed change without an anchor, a node that the input does not feed, a burst call in the wrong state. |
| `PendingFull` | `CapacityExceeded` | `settings.pending_params` changes wait already. |
| `Build(Report)` | `Validation` | The description did not validate. |
| `Json(String)`, `Yaml(String)`, `Wire(String)` | `Parse` | The text could not be parsed into the model, or a wire message is malformed. |
| `Internal` | `Internal` | An invariant of the runner was violated; cannot happen after a successful build. |

Errors are only created on the error path; the hot path never builds a string.

## JSON Schema

`Registry::json_schema()` returns the JSON Schema (draft 2020-12) of the description; `Registry::json_schema_text()` is the committed form (pretty, keys sorted, LF). `schemars` derives the static part from the Rust model; the `nodes` part is generated from the block descriptions: `nodes.additionalProperties` is a `oneOf` with one alternative per kind, each with the kind as a `const`, the parameters with type, `minimum`, `maximum`, `enum`, `default` and a description in the form `<description> Unit: <unit>. Typical: <typical>. <Build-time|Runtime>.`, and the required parameters. `$id` is `https://github.com/Andste82/sdrk/schema/sdrk-pipeline-v1.json`.

`cargo xtask schema` writes `schema/sdrk-pipeline-v1.json`, `cargo xtask schema --check` compares it with the committed file (CI runs the check). A test in `crates/pipeline/sdrk-pipeline/tests/schema.rs` regenerates the schema in memory and fails if the committed file differs.

**Versioning rule.** Any change of a block description changes the schema. A *compatible* change (a new kind, a new optional parameter, a wider range, a new option, better text) regenerates `sdrk-pipeline-v1.json`. A **breaking** change (a removed parameter, a narrowed range, a changed default, a removed option) must not change the v1 file: it requires a new schema version file `schema/sdrk-pipeline-v2.json` (and a new format version) next to the old one.

## Examples

Twenty flows in `examples/flows`, each with a YAML twin (`.yaml`, with comments), all valid against the schema and covered by the tests:

- `fm_receiver.json`: RTL-SDR IQ (`cu8`, 2.4 MHz) to a DDC, an FM demodulator and 48 kHz audio, plus an averaged spectrum frame output fed from the same input buffer.
- `spectrum_only.json`: `cu8` IQ through a DC blocker into a Blackman-Harris spectrum with 8 averages.
- `resample_audio.json`: real `f32` audio at 48 kHz converted to 44.1 kHz by the rational resampler (with `max_taps` raised, because the default of 1023 is too small for this ratio).
- `fm_receiver_grouped.json`: the FM receiver with the tuner in the group `front` and the rest in `audio`.
- `nfm_receiver.json`: narrow-band FM for two-way radio: `cu8` at 2.4 MS/s through `ddc` (decimation 50, 8 kHz cutoff, 4 kHz transition, `max_taps` raised to 4096 because the default of 1023 is too small for a 4 kHz transition at 2.4 MS/s) and the power `squelch` into `fm_demod` (5 kHz deviation, 530 us de-emphasis since M25) and the node `ctcss` (`biquad`, high-pass, order 4, 300 Hz: it removes the CTCSS sub-tone of 67 to 254 Hz) to 48 kHz audio; while the squelch is closed the audio of the `f32` flow is exactly zero from 0.2 s after the end of a transmission (the state of the high-pass decays and is flushed), in a Q15 flow `fm_demod` settles at the constant rest of the Q15 `deemphasis` dead band (-12 LSB measured, at most 16 LSB asserted, `docs/kernels.md`) and `ctcss` at a constant DC rest of 321 LSB (about -40 dBFS, inaudible), which is the approved Q15 bound of the silence check (D22); the f32 chunked run is compared by SNR (97 dB) because the f32 `deemphasis` of the SIMD backend is a tolerance kernel; the YAML twin explains the values for 12.5 kHz and 25 kHz channels.
- `fm_stereo_receiver.json`: FM broadcast in stereo: `cu8` at 2.4 MS/s through `ddc` (+600 kHz, decimation 5 to 480 kHz, cutoff 130 kHz, transition 30 kHz) and `fm_stereo_demod` (defaults, 480 kHz to 48 kHz) to a `C32` stream `audio` (left in `re`, right in `im`), plus the spectrum branch of `fm_receiver`. The channel is wider than in `fm_receiver`: a stereo station is about 256 kHz wide, and 240 kHz with a cutoff of 100 kHz would limit the separation to about 29 dB (43 dB with this channel). The `fm-receiver` example writes the output as a two-channel WAV file; `--set demod.stereo=mono` forces mono output while the flow runs (the parameter is a runtime Choice, `set_param("demod", "stereo", "mono")`).
- `fm_stereo_receiver_primitives.json`: the FM stereo receiver built from primitives: `cs16` at 480 kHz (the station at 0 Hz) through `ddc` (decimation 2, cutoff 100 kHz), `quad_demod` (75 kHz), `stereo_pll` (the multiplex signal at port `a`, the same signal through the 19 kHz band-pass `pilot`, a `biquad` with Q 5, at port `b`), `decimate` (the audio filter of `fm_stereo_demod`: 15 kHz, transition 3.75 kHz, 60 dB, by 5), `stereo_matrix` and `deemphasis` (50 us on both channels of the complex stereo audio, D396) to 48 kHz stereo `C32`. It gives the same audio as `fm_stereo_demod`, bit for bit; the band-pass has no group delay in the delay model, so the two paths need no automatic delay.
- `real_input_fm_receiver.json`: FM broadcast from a real-sampling device: `rs16` at 4.8 MS/s (an Airspy-class device; the station at a real IF of 1.8 MHz) through the automatic real `convert`, `real_to_iq` (default transition 0.1 fs and 60 dB: 4.8 MS/s real to 2.4 MS/s complex, the output centre is the input frequency 1.2 MHz), `ddc` (`frequency_hz` = station minus 1.2 MHz = +600 kHz, decimation 10, cutoff 100 kHz) and `fm_demod` (as `fm_receiver`) to 48 kHz `f32` audio; on the E2E-REAL signal it reaches more than 80 dB SINAD in `f32` and Q15 (thresholds 40 dB and 35 dB). Tune with `--set tune.frequency_hz=...` (station minus 1.2 MHz).
- `am_receiver.json`: AM broadcast (medium wave, short wave): `cu8` at 2.4 MS/s through `ddc` (decimation 50, 5 kHz cutoff, 2 kHz transition, `max_taps` raised to 8192 because the design needs about 4400 taps), the `agc` with the slow time constants for AM (attack 0.1 s, release 0.5 s) and `am_demod` (envelope, 30 Hz DC blocker) into 48 kHz audio; behind it the node `audio_filter` (`biquad`, low-pass, order 4, 4000 Hz) limits the hiss above the audio band, `cutoff_hz` can be changed while the flow runs (the node is not called `audio` because the output has that name and names are shared).
- `airband_receiver.json`: the same with the power `squelch` (-40 dBFS) between `ddc` and `agc`; the YAML twin explains the 25 kHz and 8.33 kHz channel spacings. After the gate closes the audio is exactly zero (the squelch writes zeros, the `agc` holds its gain on them and `am_demod` writes zero audio and clears its DC blocker), as in `nfm_receiver`.
- `ssb_receiver.json`: SSB voice (amateur radio, HF utility, maritime): `cu8` at 2.4 MS/s through `ddc` (decimation 50, 4 kHz cutoff, 2 kHz transition, `max_taps` raised to 8192), the power `squelch` (-50 dBFS), the `agc` with its defaults (attack 5 ms, release 0.2 s, which suit SSB) and `ssb_demod` (USB, 300 to 3000 Hz) and the node `audio_filter` (`biquad`, low-pass, order 4, 2800 Hz, `cutoff_hz` changeable while the flow runs) to `f32` audio at 48 kHz. The suppressed carrier must be at 0 Hz after the `ddc` (`--set tune.frequency_hz=...`); `--set demod.sideband=lsb` switches to the lower sideband (the YAML twin explains the convention: LSB below 10 MHz in amateur radio) and `audio_decimation` can be raised to decimate the audio.
- `hf_am_receiver.json` (M33): short-wave AM with interference: `cs16` at 192 kHz through `nb` (`noise_blank`, lookahead 10 us, no hold, D373) before the channel filter, `tune` (`ddc`, decimation 4, 4.5 kHz cutoff), `squelch` (defaults), `agc` (attack 0.1 s, release 0.5 s), `demod` (`am_demod`, audio decimation 4 to 12 kHz, 4 kHz audio cutoff) and `notch` (`auto_notch`, defaults) to 12 kHz audio. The YAML twin explains both blocks and when to remove them. End-to-end tests E2E-NB and E2E-NOTCH: `crates/testing/sdrk-testkit/tests/e2e_noise_reduction_m33.rs`; the flow test: `crates/pipeline/sdrk-pipeline/tests/noise_reduction.rs`. The flow runs in `f32` only (`auto_notch` has no Q15 variant). Measured on the chain of the flow, built from blocks with the parameters of the flow file; the test `the_chain_is_the_flow` in the same file parses `hf_am_receiver.json` and fails when its nodes, parameters or edges differ from that chain (D377, D378, D423): E2E-NB, the SINAD at 1 kHz is 23.39 dB with the blanker against 7.57 dB without (`f32`) and 23.39 dB against 9.70 dB (Q15), required +10 dB; without impulses the audio with and without the blanker is bit-identical after the delay of 2 samples. E2E-NOTCH: the 1.5 kHz whistle is attenuated by 47.1 dB after 0.5 s (required 30 dB) and the audio level in 300 to 3000 Hz changes by 0.23 dB (at most 2 dB). With the spec's first block defaults of the blanker (lookahead 50 us, hold 0.2 ms) E2E-NB missed the 10 dB, so the defaults are now the values of the flow (D425).
- `fm_transmitter.json`: 48 kHz audio to FM broadcast, 2.4 MS/s `cs8` output for pull mode: `resample` 5/1, the node `shape` (`preemph_limit`: 50 us pre-emphasis and a peak limiter at `limit` 1.0, at the rate of the modulator, M25), `fm_mod` with 75 kHz deviation, `duc` by 10 at +600 kHz. The pre-emphasis matches the 50 us de-emphasis of `fm_receiver` (the loop test TX-FLAT measures the levels from 100 Hz to 12 kHz within 0.04 dB); the limiter keeps the deviation at 0.998 times 75 kHz for overdriven noise.
- `fm_stereo_transmitter.json` (M36): 48 kHz stereo audio (`c32`, left in `re`, right in `im`) to FM stereo broadcast, 1.92 MS/s `cs8` output for pull mode (`max_chunk` 4800): `up` (`resample` to 192 kHz), `pre` (`preemph_limit`, 50 us, `limit` 1.0, one gain for both channels, after the resampler: D487), `mpx` (`fm_stereo_mod`: `main` 0.9, `pilot` 0.09 at 19 kHz, the runtime parameter `pilot` 0 switches receivers to mono), `mod` (`fm_mod`, 75 kHz) and `duc` (by 10, +600 kHz). Tests in `crates/pipeline/sdrk-pipeline/tests/stereo_tx.rs`: the flow validates without warnings, pull equals push; E2E-STEREO-TX (`e2e_fm_stereo_loop_f32`) runs the flow without `duc` into `fm_stereo_demod` at 192 kHz, and the limiter test drives it with a full-scale 15 kHz burst. Measured: the pilot is locked from 0.02 s, the levels are -0.922 dB (L, 1 kHz) and -0.900 dB (R, 3 kHz), the separation 68.6 dB and 60.9 dB, mono 0.056 s after `pilot` 0. The limiter test keeps a full-scale 15 kHz burst at a multiplex peak of 0.990 (74.2 kHz deviation); with the limiter at 48 kHz before the resampler, as the milestone first specified, it reached 1.112 (83.4 kHz), because the resampler rebuilds the peaks between the limited samples (D487).
- `frame_loop.json` (M37): the proof flow of the packet ports: the packet input `msg` (`rate_hz` 48000) through `enc` (`frame_encode`, 4 preamble bytes, idle `zero`) and `dec` (`frame_decode`, exact sync word, no bad frames) to the packets output `out` (`queue` 4096). Every pushed packet comes out unchanged with `CRC_OK`, in order, with `index` = the sample position of its sync word; the test `frame_loop_exact` checks 1000 random packets in push mode and, with the encoder in a flow of its own pulled with an idle source between the packets, in pull mode.
- `qpsk_transmitter.json` (M40): packets to QPSK at 25 kBd, 1 MS/s `cs8` output for pull mode (`max_chunk` 64 packets, D621): the packet input `msg` (`rate_hz` 50000, one bit per sample) through `enc` (`frame_encode`, 4 preamble bytes, idle `preamble`: between packets the transmitter keeps a steady carrier symbol on the air), `mod` (`psk_mod`, order 4, 8 samples per symbol, roll-off 0.35, span 10: 200 kHz), `level` (`scale`, -3 dB: the overlapping pulses peak at 1.12 in I or Q and would clip in `cs8`; at -3 dB the peak is 0.79) and `duc` (by 5, cutoff 50 kHz, transition 60 kHz: pass band to 20 kHz, stop band from 80 kHz). In pull mode an idle encoder is called again until the output moves (D620): a QPSK `psk_mod` needs two bits for one symbol.
- `qpsk_receiver.json` (M40): the counterpart of `qpsk_transmitter`: `cs16` at 1 MS/s with the signal at 0 Hz through `tune` (`ddc`, decimation 5, the filter of the transmitter's `duc`), `agc` (`reference_db` 0: the demodulator expects unit amplitude, and the default -6 dB halves the gains of its carrier loop), `demod` (`psk_demod`, order 4, 25000 Bd, `carrier_bw` 0.04, `max_offset` 0.1 cycles per symbol: a carrier 2 kHz off is 0.08 cycles per symbol, D622) and `dec` (`frame_decode`, `max_sync_errors` 2, `polarity` both) to the packets output `packets`. `polarity` both resolves a lock 180 degrees off; a lock 90 or 270 degrees off decodes nothing (the `four` option of `frame_decode` comes with M44). Test 5 of M40 (`qpsk_loop`, below) measures 998 of 1000 packets at Eb/N0 10 dB with a 2 kHz offset when the loop locks on 0 or 180 degrees.
- `nfm_transmitter.json`: narrow-band FM for two-way and amateur radio, the counterpart of `nfm_receiver`: 48 kHz audio through `voice` (`biquad`, low-pass, order 4, 3 kHz), `shape` (`preemph_limit`, 530 us = the 6 dB per octave from 300 Hz of ETSI EN 300 086, `limit` 0.85), `splatter` (`biquad`, low-pass, order 4, 3 kHz: it removes the harmonics the limiter makes), `fm_mod` (5 kHz deviation, at 48 kHz), `resample` (complex, to 240 kHz) and `duc` by 10 at +200 kHz to `cs8` at 2.4 MS/s. The `limit` of 0.85 leaves room for the overshoot of the `splatter` filter (the peak deviation stays below 0.81 times 5 kHz for overdriven noise). The YAML twin explains 12.5 kHz channels (`deviation_hz` 2500) and the offset of the `duc`.
- `am_transmitter.json` (M35): AM broadcast, the counterpart of `am_receiver`: 48 kHz audio through `lpf` (`biquad`, low-pass, order 4, 4500 Hz: the 9 kHz channel), `mod` (`am_mod`, carrier 0.5, depth 0.8: the envelope peaks at 0.9 of full scale) and `duc` by 50 at +600 kHz to `cs8` at 2.4 MS/s, in pull mode. The `duc` image filter is set to cutoff 24 kHz and transition 38.4 kHz (pass band to 4.8 kHz, stop band from 43.2 kHz, about 230 taps), because the default (cutoff 0.4 and transition 0.1 times the input rate) needs about 1800 taps at interpolation 50, more than `max_taps` 1023.
- `ssb_transmitter.json` (M35): SSB voice, the counterpart of `ssb_receiver`: 48 kHz audio through `mod` (`ssb_mod`, `usb`, band 300 to 3000 Hz, 873 taps, delay 436 samples) and `duc` by 50 with the suppressed carrier at 0 Hz (the same image filter as `am_transmitter`) to `cs8` at 2.4 MS/s, in pull mode; `sideband` can be switched while the flow runs. A tone of amplitude A comes out with amplitude A, so speech near full scale clips in `cs8`.
  Flow test (Test 6 of M35): `crates/pipeline/sdrk-pipeline/tests/am_ssb_transmitters.rs` (both validate without warnings, build in `f32` and Q15, a pull of 10^6 bytes equals a push run of the same audio in pieces of 1000 and 777 samples); the pull does not allocate (`no_alloc`). The loop tests LOOP-AM and LOOP-SSB (Tests 3 and 4) run the blocks `am_mod` into `am_demod` and `ssb_mod` into `ssb_demod` in `crates/testing/sdrk-testkit/tests/e2e_loop_m35.rs`.

## Tests

`crates/pipeline/sdrk-pipeline/tests`: `stereo` (the kind `fm_stereo_demod` and the flow `fm_stereo_receiver`: registration, cost formula, `rate_unsupported`, the pipeline equals the hand-wired blocks bit for bit for `f32` and Q15, push-size invariance, the runtime parameter `stereo`, a tunable resampler behind the decoder, the YAML twin), `model` (JSON round trip, canonical form, rates, formats), `schema` (the committed schema, examples against it with `jsonschema`), `validation` (the invalid-flow corpus, all errors at once, limits, cost and memory), `e2e` (the pipeline equals the hand-wired blocks bit for bit, block-size invariance with 10 chunk plans, fan-out, a run-time parameter change), `runner` (drop policies, errors, outputs, `set_param`, `config`, `stats`, `reset`), `no_alloc` (counting allocator, also for metadata, events, timed parameters and `pull`), `property` (random graphs, random bytes and JSON values), `time_events` (gaps, anchors, joining, events, timed parameters), `pull` (transmit, underrun, bursts), `groups` (groups, backend preferences, `split_by_groups`), `yaml`, `wire` (round trips, the committed bytes in `tests/wire`, properties) and `probe_two_threads` (a split pipeline on two threads equals the single pipeline) and `nfm` (the `nfm_receiver` flow gates a narrow-band FM burst: exact silence before the transmission and from 0.2 s after its end, the Q15 flow with `fm_demod` exactly zero and `ctcss` at its recorded offset, chunk invariance, run-time threshold) and `am` (the `am_receiver` and `airband_receiver` flows on the E2E-AM signal: SINAD and audio level, the airband burst, chunk invariance, run-time parameters of `agc` and `squelch`, the cost of the new kinds) and `biquad` (the cost formula, memory, delay, invalid parameters, run-time `cutoff_hz` and `q`, the audio filters of the three flows) and `halfband_decimate` (the kinds `halfband_decimate` and `real_to_iq`: registration, the cost `(N + 3) / 4`, rates and delay, parameter checks, the real automatic conversion, the `real_input_fm_receiver` flow on the E2E-REAL signal in `f32` and Q15, chunk invariance, the YAML twin) and `one_input_kinds` (the kinds `to_complex`, `real_part`, `imag_part`, `nco` and `stereo_matrix`: registration with their kernels, the costs 1, 1, 1, 6 and 2, type and parameter checks, `real_part` of `to_complex` returns the input, `nco` driven by 1 kHz audio gives the complex tone, the run-time `frequency_hz`) and `ssb` (the `ssb_receiver` flow: validation, the cost formula, SINAD, the run-time switch of the sideband, chunk invariance) and `primitive_kinds` (the seven kinds `quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`, `mag` and `delay`: registration and kernel names, the cost formulas, output rates, the delay figures, the plan estimate against the built pipeline, type and parameter checks, and the flow `rotate`, `decimate`, `quad_demod`, `deemphasis`, `decimate` (real) against `fm_receiver`). and `multi_input` (the edge field `port` and the field `align`: the model, every new diagnostic with its path and message, the four kinds, the automatic `delay` (a 31-tap `fir` on one path gives `<node>__align` with 15 samples), `align: none`, `add` of two paths equals the kernel, chunk-size invariance with sizes 1, 7 and 4096 and 20 chunk plans of seed 478, pull mode equals push mode, bursts reset both paths, the buffers of the two ports stay bounded over 10^7 samples, and the proof flow `fm_stereo_receiver_primitives` equals `fm_stereo_demod` bit for bit, with the de-emphasis and, without its node, with `deemphasis_us` 0) and `packets` (M37: the flow `frame_loop` with its buffer and memory figures, `frame_loop_exact` in push mode with 1000 random packets of 0 to 253 bytes in random groups (seed 479) and in pull mode with an idle source for both idle signals, bursts behind a packet input, the encoder without packets in push mode, `drop_oldest`, `drop_newest` and `error` on a full packet queue, fan-out to two encoders and a direct output, wrong calls, `reset`, the diagnostics of packet edges and outputs, a packet input without `rate_hz`, the clock domain of a packet input, packet edges between groups). The unit tests in `src/clocked_test.rs` register a test kind with a `Clocked<f32, Packets>` input (clock pace, idle, waiting packets, the warning for an ignored `rate_hz`, `type_mismatch` at port `b`, `reset`); `no_alloc` covers the packet flow in push and pull mode. The fuzz target `fuzz_targets/pipeline_json.rs` (`cargo +nightly fuzz run pipeline_json` in `crates/pipeline/sdrk-pipeline`) also builds a small flow with a node of two inputs from its first bytes (kind, ports, `align`, a filter on one path, the rate of the second input) and runs in the nightly CI job for five minutes on Linux x86_64 only; fuzzing is not required on Windows and arm64.
`crates/pipeline/sdrk-pipeline/tests`: `am_ssb_transmitters` (M35: the flows `am_transmitter` and `ssb_transmitter`: validation, `f32` and Q15 builds, pull equals push, the parameters of the spec), `am_ssb_mod_kinds` (M35: the kinds `am_mod` and `ssb_mod`: registration with their kernels, the costs 2 and `8 + 4 * taps`, the delay `(taps - 1) / 2`, type and parameter checks with `type_mismatch` for complex input, the runtime parameters `depth` and `sideband`, a push run equals the block), `noise_reduction` (M33: the kinds `noise_blank` and `auto_notch`: registration, costs, delays, type checks with `type_mismatch` for Q15 into `auto_notch`, runtime parameters on a running pipeline, and the flow `hf_am_receiver`: validates without warnings, builds and runs in push mode), `stereo` (the kind `fm_stereo_demod` and the flow `fm_stereo_receiver`: registration, cost formula, `rate_unsupported`, the pipeline equals the hand-wired blocks bit for bit for `f32` and Q15, push-size invariance, the runtime parameter `stereo`, a tunable resampler behind the decoder, the YAML twin), `model` (JSON round trip, canonical form, rates, formats), `schema` (the committed schema, examples against it with `jsonschema`), `validation` (the invalid-flow corpus, all errors at once, limits, cost and memory), `e2e` (the pipeline equals the hand-wired blocks bit for bit, block-size invariance with 10 chunk plans, fan-out, a run-time parameter change), `runner` (drop policies, errors, outputs, `set_param`, `config`, `stats`, `reset`), `no_alloc` (counting allocator, also for metadata, events, timed parameters and `pull`), `property` (random graphs, random bytes and JSON values), `time_events` (gaps, anchors, joining, events, timed parameters), `pull` (transmit, underrun, bursts), `groups` (groups, backend preferences, `split_by_groups`), `yaml`, `wire` (round trips, the committed bytes in `tests/wire`, properties) and `probe_two_threads` (a split pipeline on two threads equals the single pipeline) and `nfm` (the `nfm_receiver` flow gates a narrow-band FM burst: exact silence before the transmission and from 0.2 s after its end, the Q15 flow with `fm_demod` exactly zero and `ctcss` at its recorded offset, chunk invariance, run-time threshold) and `am` (the `am_receiver` and `airband_receiver` flows on the E2E-AM signal: SINAD and audio level, the airband burst, chunk invariance, run-time parameters of `agc` and `squelch`, the cost of the new kinds) and `biquad` (the cost formula, memory, delay, invalid parameters, run-time `cutoff_hz` and `q`, the audio filters of the three flows) and `halfband_decimate` (the kinds `halfband_decimate` and `real_to_iq`: registration, the cost `(N + 3) / 4`, rates and delay, parameter checks, the real automatic conversion, the `real_input_fm_receiver` flow on the E2E-REAL signal in `f32` and Q15, chunk invariance, the YAML twin) and `one_input_kinds` (the kinds `to_complex`, `real_part`, `imag_part`, `nco` and `stereo_matrix`: registration with their kernels, the costs 1, 1, 1, 6 and 2, type and parameter checks, `real_part` of `to_complex` returns the input, `nco` driven by 1 kHz audio gives the complex tone, the run-time `frequency_hz`) and `ssb` (the `ssb_receiver` flow: validation, the cost formula, SINAD, the run-time switch of the sideband, chunk invariance) and `primitive_kinds` (the seven kinds `quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`, `mag` and `delay`: registration and kernel names, the cost formulas, output rates, the delay figures, the plan estimate against the built pipeline, type and parameter checks, and the flow `rotate`, `decimate`, `quad_demod`, `deemphasis`, `decimate` (real) against `fm_receiver`). The fuzz target `fuzz_targets/pipeline_json.rs` (`cargo +nightly fuzz run pipeline_json` in `crates/pipeline/sdrk-pipeline`) runs in the nightly CI job for five minutes on Linux x86_64 only; fuzzing is not required on Windows and arm64.
`crates/pipeline/sdrk-pipeline/tests` (M38): `symbol_kinds` (the kinds `symbol_sync` and `slice`: registration (`symbol_sync` with its kernel, `slice` without one; 45 kinds), the costs `2 + 24 / sps` and 1, the output rates (the symbol rate; the symbol rate times the bits per symbol) and the delays (1 input sample; 0), the required `symbol_rate_hz`, exactly 2 samples per symbol accepted and fewer `rate_unsupported` at `params.symbol_rate_hz`, the parameter ranges, `type_mismatch` for `q15` and `cq15`, `invalid_value` at `params.constellation` for a constellation that does not take the input type, and a push-mode chain `symbol_sync` -> `slice` in pieces of up to 1000 samples that equals the blocks, with a run-time change of `scale` and a rejected change of the build parameter `loop_bw`) and `bpsk_sync` (test 7: the test flow `tests/flows/bpsk_sync_test.json`, not an example: a `c32` input at 41 kHz, `symbol_sync` at 10000 Bd with Gardner and the block defaults (D565), `slice` `bpsk` and `frame_decode` into a packets output. A `frame_encode` pipeline makes 100 frames, the test shapes them as raised-cosine BPSK (0.35, 4.1 samples per symbol, 1024 alternating training symbols before and 64 after) and sends them through 50 ppm clock drift and noise at Eb/N0 20 dB (seed 493, D559); all 100 packets come out with `CRC_OK`, unchanged, in order, with sync-word indices exactly one frame apart (no symbol lost or inserted) and nothing dropped).
`crates/pipeline/sdrk-pipeline/tests` (M40): `psk_kinds` (the kinds `costas`, `psk_mod` and `psk_demod`: registration with their kernels (48 kinds), the costs 30, `4 span` and `4 span + 40 / sps`, the output rates and delays (`span bits / 2` input samples for `psk_mod`, `span n / 2 + 1` for `psk_demod`), the required `samples_per_symbol` and `symbol_rate_hz`, `rate_unsupported` at `params.symbol_rate_hz` with the hint to resample when the rate is not within 1 % of an integer number of samples per symbol, `invalid_value` at `params.order` for the order 3, the type checks, the parameter `polarity` of `frame_decode`, push-mode `psk_mod` and `psk_demod` in pieces of 1 to 1024 samples (seed 559) equal to the blocks bit for bit, and BPSK frames from `frame_encode` -> `psk_mod` through a channel that inverts the carrier: no packet with `polarity` `normal`, all of them with `both`) and `qpsk_loop` (test 5: the flows `qpsk_transmitter` and `qpsk_receiver` validate without warnings (rates 200 kHz and 1 MHz; 200 kHz and 50 kbit/s); one-sample pulls of the idle transmitter equal one large pull (D620); 1000 packets of 0 to 63 bytes after 20000 idle symbols, pulled from the transmitter, shifted by 2 kHz and given noise at Eb/N0 10 dB (seed 558), into the receiver as `cs16`, which runs on the channel output rotated by 0, 90, 180 and 270 degrees in four threads: at least 995 packets with `CRC_OK` and unchanged payload for the two rotations that put the lock on 0 or 180 degrees, the same packets for rotations 180 degrees apart, none for the other two; measured 998 of 1000, about 20 s in the test profile; D623).
`crates/pipeline/sdrk-pipeline/tests`: `stereo_tx` (M36: the kinds `preemph_limit` with `c32` input and `fm_stereo_mod`: registration with their kernels, the costs 12 and 10, rates and delays, `type_mismatch` for `cq15` into `preemph_limit` and for every type but `c32` into `fm_stereo_mod`, `rate_unsupported` below 120 kHz, parameter ranges, the runtime parameters `limit` and `pilot`, and the pipeline of both kinds equals the two blocks bit for bit with a parameter change in the middle; the flow `fm_stereo_transmitter` validates without warnings and its pull output equals the push output; E2E-STEREO-TX `e2e_fm_stereo_loop_f32` (seed 488) and the limiter test (seed 489) on the flow), `noise_reduction` (M33: the kinds `noise_blank` and `auto_notch`: registration, costs, delays, type checks with `type_mismatch` for Q15 into `auto_notch`, runtime parameters on a running pipeline, and the flow `hf_am_receiver`: validates without warnings, builds and runs in push mode), `stereo` (the kind `fm_stereo_demod` and the flow `fm_stereo_receiver`: registration, cost formula, `rate_unsupported`, the pipeline equals the hand-wired blocks bit for bit for `f32` and Q15, push-size invariance, the runtime parameter `stereo`, a tunable resampler behind the decoder, the YAML twin), `model` (JSON round trip, canonical form, rates, formats), `schema` (the committed schema, examples against it with `jsonschema`), `validation` (the invalid-flow corpus, all errors at once, limits, cost and memory), `e2e` (the pipeline equals the hand-wired blocks bit for bit, block-size invariance with 10 chunk plans, fan-out, a run-time parameter change), `runner` (drop policies, errors, outputs, `set_param`, `config`, `stats`, `reset`), `no_alloc` (counting allocator, also for metadata, events, timed parameters and `pull`), `property` (random graphs, random bytes and JSON values), `time_events` (gaps, anchors, joining, events, timed parameters), `pull` (transmit, underrun, bursts), `groups` (groups, backend preferences, `split_by_groups`), `yaml`, `wire` (round trips, the committed bytes in `tests/wire`, properties) and `probe_two_threads` (a split pipeline on two threads equals the single pipeline) and `nfm` (the `nfm_receiver` flow gates a narrow-band FM burst: exact silence before the transmission and from 0.2 s after its end, the Q15 flow with `fm_demod` exactly zero and `ctcss` at its recorded offset, chunk invariance, run-time threshold) and `am` (the `am_receiver` and `airband_receiver` flows on the E2E-AM signal: SINAD and audio level, the airband burst, chunk invariance, run-time parameters of `agc` and `squelch`, the cost of the new kinds) and `biquad` (the cost formula, memory, delay, invalid parameters, run-time `cutoff_hz` and `q`, the audio filters of the three flows) and `halfband_decimate` (the kinds `halfband_decimate` and `real_to_iq`: registration, the cost `(N + 3) / 4`, rates and delay, parameter checks, the real automatic conversion, the `real_input_fm_receiver` flow on the E2E-REAL signal in `f32` and Q15, chunk invariance, the YAML twin) and `one_input_kinds` (the kinds `to_complex`, `real_part`, `imag_part`, `nco` and `stereo_matrix`: registration with their kernels, the costs 1, 1, 1, 6 and 2, type and parameter checks, `real_part` of `to_complex` returns the input, `nco` driven by 1 kHz audio gives the complex tone, the run-time `frequency_hz`) and `ssb` (the `ssb_receiver` flow: validation, the cost formula, SINAD, the run-time switch of the sideband, chunk invariance) and `primitive_kinds` (the seven kinds `quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`, `mag` and `delay`: registration and kernel names, the cost formulas, output rates, the delay figures, the plan estimate against the built pipeline, type and parameter checks, and the flow `rotate`, `decimate`, `quad_demod`, `deemphasis`, `decimate` (real) against `fm_receiver`). The fuzz target `fuzz_targets/pipeline_json.rs` (`cargo +nightly fuzz run pipeline_json` in `crates/pipeline/sdrk-pipeline`) runs in the nightly CI job for five minutes on Linux x86_64 only; fuzzing is not required on Windows and arm64.
