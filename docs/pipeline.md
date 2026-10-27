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

An input is either raw bytes in a sample `format` or already converted samples of a `type`. Exactly one of the two is given.

| Field | Type | Meaning |
| --- | --- | --- |
| `format` | name or object | Raw sample format (below). An automatic `convert` node follows the input. |
| `type` | `"c32"`, `"cq15"`, `"f32"`, `"q15"` | Converted samples. The type must fit the precision: `c32` and `f32` with `"f32"`, `cq15` and `q15` with `"q15"`. |
| `rate_hz` | integer or `"num/den"` | Sample rate in Hz. A positive integer, or an exact fraction written as a string; it is reduced on load (`"96000/2"` becomes `48000`) and written as an integer if it is whole. |
| `description` | string, optional | |

**Format names:** `cu8`, `cs8`, `cs16`, `cs12_lsb16`, `sc16_q11`, `cs16_planar`, `packed12`, `packed10`, `i2s32_msb`, `cf32`, `rf32`, `rs16` map to the constants of `sdrk_device::formats`. A custom format is an object `{ "container": "i16", "bits": 12, "align": "lsb", "layout": "interleaved", "endian": "le", "full_scale": "power_of_two" }` with `container` one of `u8 i8 i16 i32 f32 packed12 packed10 bits1`, `align` `lsb`/`msb`, `layout` `interleaved`/`planar`/`real`, `endian` `le`/`be`, `full_scale` `"power_of_two"` or `{"explicit": 2048}` (optional) and an optional `swap_iq` flag. The object is validated by `sdrk-device`.

A raw input must be a streamable complex format: interleaved layout, no 1-bit container. Planar and real formats (`cs16_planar`, `rf32`, `rs16`) are rejected at `inputs.<name>.format`; feed real audio as `"type": "f32"` instead.

### Nodes

| Field | Type | Meaning |
| --- | --- | --- |
| `kind` | string | A block kind of the registry, e.g. `"ddc"`. The kinds, their ports and all parameters are listed in [blocks.md](blocks.md) and in the schema. |
| `params` | object, optional | Parameters by name. |
| `group` | string, optional | A key of `groups`; nodes without one are in the implicit group `"main"`. |
| `backend` | object, optional | `{"prefer": name}` or `{"require": name}`, see [Groups](#groups-and-backend-preferences). |
| `description` | string, optional | |

A parameter value is a JSON number, a boolean or a string. Integers must be exact: `10` and `10.0` are accepted for an integer parameter, `10.5` is not. A `Choice` parameter (for example the `window` of `spectrum`) is given by its option name, a `Format` parameter (`convert`, `convert_out`) by a format name or object. A parameter that is not given takes the default of the description or the rate-dependent default of the block (for example the cutoff of the `ddc`). Parameters listed under `required` in the description (for example `decimation` of `ddc`, `rate_out_hz` of `resampler`) must be given.

Parameters are either **build-time** (fixed when the pipeline is built) or **runtime** (changed with `Pipeline::set_param`, see below). The description of each parameter says which.

### Edges

`{ "from": "<input or node>", "to": "<node or output>" }`. Every node and every output has exactly one incoming edge. A source may have any number of outgoing edges (fan-out; all readers share one buffer, no copy). The graph must be acyclic. The output of every node must be connected to a node or an output. An input without an outgoing edge is only a warning.

An input may be connected directly to an output.

### Outputs

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | `"stream"` or `"frames"` | `"stream"` takes the output of a block, `"frames"` the output of a frame block (`spectrum`). The type must match what the source produces. |
| `format` | name or object, optional | Stream outputs only: convert to raw bytes in this format (an automatic `convert_out` node). Only complex streams can be converted. |
| `capacity` | integer, optional | Stream outputs only: size of the output queue in samples (bytes for a raw output), 1 to 2^28. Default `4 * max_chunk`. |
| `queue` | integer, optional | Frames outputs only: length of the queue in frames, 1 to 65 536. Default 4. |
| `drop` | `"error"`, `"drop_oldest"`, `"drop_newest"`, optional | What happens when the output is full. Default `"error"`. |
| `underrun` | `"zeros"`, `"short"`, optional | Pull mode: what `pull` does when the source is exhausted. Default `"zeros"` for raw outputs, `"short"` otherwise. |
| `description` | string, optional | |

Drop policies:

| Policy | Behaviour |
| --- | --- |
| `error` | `push` fails with `OutputFull` **before it consumes anything** when the worst-case result of the push (`ceil(len * R)` elements, with `R` the rate ratio from the input to the output, one more for rounding; capped at the capacity of the output) does not fit. Nothing is lost. If a block still produces more than estimated, the blocked nodes keep their data until the reader makes room (back-pressure); reading from the output continues the processing. |
| `drop_oldest` | The oldest data of the output is discarded to make room. |
| `drop_newest` | Data that does not fit is discarded. |

`Pipeline::dropped(output)` counts the discarded elements (frames for a frames output).

### Rust types

`PipelineSpec`, `Settings`, `InputSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `GroupSpec`, `BackendSpec`, `Precision`, `DropPolicy`, `UnderrunPolicy` (and the helper types `RateHz`, `FormatSpec`, `FormatName`, `CustomFormat`, `InputType`, `OutputType`) are plain `serde` structs with `#[serde(deny_unknown_fields)]`.

```rust
let spec = PipelineSpec::from_json(text)?;   // Err(Error::Json) with the position of a problem
let canonical = spec.to_json();               // pretty, keys in field order, maps sorted
```

`to_json(from_json(x))` parses back to an equal spec, and the files in `examples/flows` are canonical (the bytes are identical). Duplicate keys inside one JSON object are not detected (the last one wins), because `serde_json` does not report them.

## Registry and factories

`Registry::standard()` registers every block of `sdrk-blocks`. Each kind has a `BlockFactory`: it parses and checks the parameters for the input type, rate and precision (`check`), names the kernels the block calls (`kernels`), and builds the type-erased `Node` (`build`). `Checked` carries the figures of a checked node: output rate, memory, cost, output kind, number of taps, group delay and the smallest input count that produces an output.

The runner works on type-erased nodes: `NodeInput`/`NodeOutput` are typed, borrowed views, `StreamNode<B: Block>` and `FrameNode<B: FrameBlock>` wrap the generic blocks. A variant mismatch inside `process` is an `Internal` error that cannot happen after a successful build.

## Validation

`Registry::validate(&spec, &opts)` and `Registry::validate_json(text, &opts)` return a `Report` with **all** problems at once. Nothing is built; the figures are estimates from the configurations.

Validation has two stages:

1. **Structure.** For JSON text: syntax (with line and column), unknown fields, JSON types, enumerations, required fields, the schema version. Then, on the typed description: names, descriptions, kinds, parameter names, types and absolute ranges, output attributes.
2. **Graph.** Endpoints, one incoming edge per node and output, dangling outputs, cycles (Kahn's algorithm; one node of each cycle is reported), the port types at every edge, then the rates in topological order with exact `Rate` arithmetic and the rate-dependent checks of every block, and the limits.

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

`path` uses dots and the JSON names: `nodes.tune.params.frequency_hz`, `edges[2].to`, `inputs.rf.format`. The node report lists name, kind, input and output rate (an integer, or `"num/den"`), the group delay in input samples (`"num/den"`) and whether the node is automatic. `outputs` has one entry per output with its `group_delay` (in samples of the output, `"num/den"`) and `max_buffered` (see [timing.md](timing.md#latency)). `group_edges` (only present with groups) lists the edges between nodes of different groups.

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
| `multiple_inputs` | 2 | A node or output with more than one incoming edge. | `edges[i].to` of the extra edge |
| `missing_input` | 2 | A node or output without an incoming edge. | `nodes.<name>` / `outputs.<name>` |
| `dangling_output` | 2 | The output of a node goes nowhere. (Warning: an input nobody reads.) | `nodes.<name>` / `inputs.<name>` |
| `cycle` | 2 | A node on a cycle. | `nodes.<name>` |
| `type_mismatch` | 2 | Port types that do not fit: a block fed with the wrong sample type, frames into a node, a stream output on frames or the other way round, an input type that does not fit the precision. | `edges[i].to`, `outputs.<name>.type`, `inputs.<name>.type` |
| `rate_unsupported` | 2 | The block rejects its configuration for the sample rate at its input (frequency beyond Nyquist, a filter that does not fit, ...). | `nodes.<name>.params.<param>` |
| `limit_exceeded` | 2 | A resource limit is exceeded. | `nodes.<name>` (taps), `nodes` (the rest) |
| `unknown_group` | 1 | A node names a group that `groups` does not declare. | `nodes.<name>.group` |
| `backend_unavailable` | 2 | A required backend does not implement the kernels of the node (an error for `require`, a warning for `prefer`). | `nodes.<name>.backend.require` / `.prefer` |

The invalid-flow corpus in `crates/pipeline/sdrk-pipeline/tests/invalid` has one file per code, with the expected diagnostic in a sibling `.expected.json`.

## Build

`Registry::build(&spec, &opts)` validates first and returns the report as the error if there are errors.

- **Automatic conversions.** An input with a raw `format` gets a node `<input>__convert` (kind `convert`), and every edge from that input reads the converted stream. A stream output with a `format` gets `<output>__convert_out` (kind `convert_out`). Automatic nodes appear in the report (`auto: true`) and in the statistics, but not in the readout of `config()`, which reproduces the user's description.
- **Order.** Topological order; ties are broken by the first appearance of a node in `edges`. The automatic input converters come first, the output converters last.
- **Buffers.** One linear buffer per node output and per input, shared by all readers: each reader has its own position (zero-copy fan-out). The capacity of the buffer after node `k` is `ceil(max_chunk * R_k) + min_input + 64` elements, where `R_k` is the rate of the buffer divided by the rate of its input (exact; times the bytes per sample for a raw buffer) and `min_input` is the largest `min_input` of the readers. A writer compacts the buffer (moves unread data to the front) when the free tail is smaller than what the node could write, so every block sees contiguous input. Every output has its own queue behind the buffer, which is where the drop policies apply.
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

### Cost formulas

The cost is an estimate, in multiply-accumulates (a complex tap counts like the formulas below say), not a measurement. Each factory gives the cost **per input sample of the node**:

| Kind | Cost |
| --- | --- |
| `convert`, `convert_out` | 2 |
| `scale` | 1 |
| `dc_block`, `squelch` | 3 |
| `agc` | 6 |
| `fir` | `taps` |
| `ddc` | `4 + 4 * taps / decimation` |
| `duc` | `4 * interpolation + 4 * taps` |
| `fm_demod` | `10 + taps / audio_decimation` (`taps` of the audio filter, 0 without it) |
| `am_demod` | `4 + taps / audio_decimation` (`taps` of the audio filter, 0 without it) |
| `ssb_demod` | `8 + 4 * taps / audio_decimation` (two NCOs and the complex filter, as `ddc`) |
| `biquad` | `5 * sections` (real input) or `10 * sections` (complex input); `sections = order / 2` for low-pass and high-pass, 1 for band-pass and notch |
| `fm_mod` | 6 |
| `resampler`, rational | `taps / M`, `M` the denominator of `rate_out / rate_in` |
| `resampler`, tunable | `K * rate_out / rate_in + 4`, `K` taps per phase |
| `spectrum` | `(2.5 * N * log2(N) + 3 * N) / interval_samples`, `N = fft_size` |

The cost of the pipeline is `sum(node cost * samples per second at the node) / sum(input samples per second)`: MACs per pipeline input sample, with a node behind a decimator counted at the lower rate.

## Runner (push mode)

```rust
let mut pipeline = registry.build(&spec, &BuildOptions::default())?;
pipeline.push("rf", InputData::Raw(&bytes))?;           // up to max_chunk samples
let n = pipeline.read_stream("audio", &mut audio)?;     // T must match the output type
while let Some(meta) = pipeline.read_frame("spectrum", &mut frame)? { /* ... */ }
pipeline.set_param("tune", "frequency_hz", serde_json::json!(610_000.0))?;
```

- **`push`.** Feeds up to `max_chunk` samples to one input and runs the graph until no node can make progress. All of `data` is consumed, or an error is returned before anything is consumed. When the buffer of the input is empty, the first readers process the caller's slice directly; only what they did not take (history-limited `min_input`, a split packed sample) is copied into the input buffer, once.
- **`read_stream::<T>`, `read_raw`, `read_frame`, `available`.** Read from the output queues. `read_stream` is for `C32`, `CQ15`, `f32` and `Q15` outputs, `read_raw` for outputs in a raw format, `read_frame` for frame outputs (the buffer must have the frame length). `available` counts samples (bytes for a raw output) or frames.
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

`pull` does not allocate; `end_burst` does (the zero samples).

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
- `split_by_groups(&spec, &registry)` returns one description per group, sorted by group name. A cut edge `a -> b` becomes the output `edge__a__b` (a stream of the type of `a`) in the group of `a` and the input `edge__a__b` (the type and the exact rate) in the group of `b`. (The name rule does not allow a leading underscore, so the prefix is `edge__`.) Pipeline inputs are copied into every part that reads them.

## YAML

With the feature `yaml` (default; the WASM binding disables it) `PipelineSpec::from_yaml(text)` reads the same model from YAML, using `serde-saphyr` (version 2.0.0). Unknown fields are errors as in JSON, comments are allowed and errors carry line and column. `Registry::validate_yaml` and `Registry::build_yaml` give the same diagnostics (same codes, same paths) as the JSON functions; JSON text is also valid YAML. There is no YAML writer: JSON is canonical. Every example flow has a `.yaml` twin with comments.

## Wire format

[wire-format.md](wire-format.md) specifies the versioned messages (`wire` module) that carry samples, frames, events, stream descriptions and parameter changes between the parts of a split pipeline.

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

Eight flows in `examples/flows`, each with a YAML twin (`.yaml`, with comments), all valid against the schema and covered by the tests:

- `fm_receiver.json`: RTL-SDR IQ (`cu8`, 2.4 MHz) to a DDC, an FM demodulator and 48 kHz audio, plus an averaged spectrum frame output fed from the same input buffer.
- `spectrum_only.json`: `cu8` IQ through a DC blocker into a Blackman-Harris spectrum with 8 averages.
- `resample_audio.json`: real `f32` audio at 48 kHz converted to 44.1 kHz by the rational resampler (with `max_taps` raised, because the default of 1023 is too small for this ratio).
- `fm_receiver_grouped.json`: the FM receiver with the tuner in the group `front` and the rest in `audio`.
- `nfm_receiver.json`: narrow-band FM for two-way radio: `cu8` at 2.4 MS/s through `ddc` (decimation 50, 8 kHz cutoff, 4 kHz transition, `max_taps` raised to 4096 because the default of 1023 is too small for a 4 kHz transition at 2.4 MS/s) and the power `squelch` into `fm_demod` (5 kHz deviation) and the node `ctcss` (`biquad`, high-pass, order 4, 300 Hz: it removes the CTCSS sub-tone of 67 to 254 Hz) to 48 kHz audio; while the squelch is closed the audio of the `f32` flow is exactly zero from 0.2 s after the end of a transmission (the state of the high-pass decays and is flushed), in a Q15 flow `fm_demod` is exactly zero and `ctcss` settles at a constant DC rest of 321 LSB (about -40 dBFS, inaudible), which is the approved Q15 bound of the silence check (D22); the YAML twin explains the values for 12.5 kHz and 25 kHz channels.
- `am_receiver.json`: AM broadcast (medium wave, short wave): `cu8` at 2.4 MS/s through `ddc` (decimation 50, 5 kHz cutoff, 2 kHz transition, `max_taps` raised to 8192 because the design needs about 4400 taps), the `agc` with the slow time constants for AM (attack 0.1 s, release 0.5 s) and `am_demod` (envelope, 30 Hz DC blocker) into 48 kHz audio; behind it the node `audio_filter` (`biquad`, low-pass, order 4, 4000 Hz) limits the hiss above the audio band, `cutoff_hz` can be changed while the flow runs (the node is not called `audio` because the output has that name and names are shared).
- `airband_receiver.json`: the same with the power `squelch` (-40 dBFS) between `ddc` and `agc`; the YAML twin explains the 25 kHz and 8.33 kHz channel spacings. After the gate closes the audio is exactly zero (the squelch writes zeros, the `agc` holds its gain on them and `am_demod` writes zero audio and clears its DC blocker), as in `nfm_receiver`.
- `ssb_receiver.json`: SSB voice (amateur radio, HF utility, maritime): `cu8` at 2.4 MS/s through `ddc` (decimation 50, 4 kHz cutoff, 2 kHz transition, `max_taps` raised to 8192), the power `squelch` (-50 dBFS), the `agc` with its defaults (attack 5 ms, release 0.2 s, which suit SSB) and `ssb_demod` (USB, 300 to 3000 Hz) and the node `audio_filter` (`biquad`, low-pass, order 4, 2800 Hz, `cutoff_hz` changeable while the flow runs) to `f32` audio at 48 kHz. The suppressed carrier must be at 0 Hz after the `ddc` (`--set tune.frequency_hz=...`); `--set demod.sideband=lsb` switches to the lower sideband (the YAML twin explains the convention: LSB below 10 MHz in amateur radio) and `audio_decimation` can be raised to decimate the audio.
- `fm_transmitter.json`: 48 kHz audio to FM, 2.4 MS/s `cs8` output for pull mode: `resampler` 5/1, `fm_mod` with 75 kHz deviation, `duc` by 10 at +600 kHz.

## Tests

`crates/pipeline/sdrk-pipeline/tests`: `model` (JSON round trip, canonical form, rates, formats), `schema` (the committed schema, examples against it with `jsonschema`), `validation` (the invalid-flow corpus, all errors at once, limits, cost and memory), `e2e` (the pipeline equals the hand-wired blocks bit for bit, block-size invariance with 10 chunk plans, fan-out, a run-time parameter change), `runner` (drop policies, errors, outputs, `set_param`, `config`, `stats`, `reset`), `no_alloc` (counting allocator, also for metadata, events, timed parameters and `pull`), `property` (random graphs, random bytes and JSON values), `time_events` (gaps, anchors, joining, events, timed parameters), `pull` (transmit, underrun, bursts), `groups` (groups, backend preferences, `split_by_groups`), `yaml`, `wire` (round trips, the committed bytes in `tests/wire`, properties) and `probe_two_threads` (a split pipeline on two threads equals the single pipeline) and `nfm` (the `nfm_receiver` flow gates a narrow-band FM burst: exact silence before the transmission and from 0.2 s after its end, the Q15 flow with `fm_demod` exactly zero and `ctcss` at its recorded offset, chunk invariance, run-time threshold) and `am` (the `am_receiver` and `airband_receiver` flows on the E2E-AM signal: SINAD and audio level, the airband burst, chunk invariance, run-time parameters of `agc` and `squelch`, the cost of the new kinds) and `biquad` (the cost formula, memory, delay, invalid parameters, run-time `cutoff_hz` and `q`, the audio filters of the three flows) and `ssb` (the `ssb_receiver` flow: validation, the cost formula, SINAD, the run-time switch of the sideband, chunk invariance). The fuzz target `fuzz_targets/pipeline_json.rs` (`cargo +nightly fuzz run pipeline_json` in `crates/pipeline/sdrk-pipeline`) runs in the nightly CI job for five minutes on Linux x86_64 only; fuzzing is not required on Windows and arm64.
