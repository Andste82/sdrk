# Pipeline

The crate `sdrk-pipeline` (layer `crates/pipeline`, heap allowed) turns a description of a signal-processing graph into a running pipeline. The description is data: JSON is the canonical format, so host languages and AI agents never write code to describe processing. The pipeline is the only place with heap: it allocates once while building and then runs the heap-free blocks of [`sdrk-blocks`](blocks.md) deterministically. **Nothing allocates after `build()`** (tested with a counting global allocator in `tests/no_alloc.rs`).

This page covers format version 1 and the push-mode runner. Events, gaps, time anchors, pull mode and transmit are added by M11.

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
| `nodes` | object | Processing nodes by name. |
| `edges` | array | Connections. |
| `outputs` | object | Outputs by name. |

`settings`:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `precision` | `"f32"` or `"q15"` | `"f32"` | `"f32"` uses complex `C32` and real `f32`, `"q15"` uses `CQ15` and `Q15`. One precision per pipeline. |
| `max_chunk` | integer 1 to 1 048 576 | 4096 | Largest number of samples accepted per `push` and input. For a raw input the limit in bytes is `max_chunk` times the bytes of one sample, rounded up. |

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
| `description` | string, optional | |

Drop policies:

| Policy | Behaviour |
| --- | --- |
| `error` | `push` fails with `OutputFull` **before it consumes anything** when the worst-case result of the push (`ceil(len * R)` elements, with `R` the rate ratio from the input to the output, one more for rounding; capped at the capacity of the output) does not fit. Nothing is lost. If a block still produces more than estimated, the blocked nodes keep their data until the reader makes room (back-pressure); reading from the output continues the processing. |
| `drop_oldest` | The oldest data of the output is discarded to make room. |
| `drop_newest` | Data that does not fit is discarded. |

`Pipeline::dropped(output)` counts the discarded elements (frames for a frames output).

### Rust types

`PipelineSpec`, `Settings`, `InputSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `Precision`, `DropPolicy` (and the helper types `RateHz`, `FormatSpec`, `FormatName`, `CustomFormat`, `InputType`, `OutputType`) are plain `serde` structs with `#[serde(deny_unknown_fields)]`.

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
  "nodes": []
}
```

`path` uses dots and the JSON names: `nodes.tune.params.frequency_hz`, `edges[2].to`, `inputs.rf.format`. The node report lists name, kind, input and output rate (an integer, or `"num/den"`), the group delay in input samples (`"num/den"`) and whether the node is automatic.

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
| `dc_block` | 3 |
| `fir` | `taps` |
| `ddc` | `4 + 4 * taps / decimation` |
| `duc` | `4 * interpolation + 4 * taps` |
| `fm_demod` | `10 + taps / audio_decimation` (`taps` of the audio filter, 0 without it) |
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

### Errors

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
| `Build(Report)` | `Validation` | The description did not validate. |
| `Json(String)` | `Parse` | The text could not be parsed into the model. |
| `Internal` | `Internal` | An invariant of the runner was violated; cannot happen after a successful build. |

Errors are only created on the error path; the hot path never builds a string.

## JSON Schema

`Registry::json_schema()` returns the JSON Schema (draft 2020-12) of the description; `Registry::json_schema_text()` is the committed form (pretty, keys sorted, LF). `schemars` derives the static part from the Rust model; the `nodes` part is generated from the block descriptions: `nodes.additionalProperties` is a `oneOf` with one alternative per kind, each with the kind as a `const`, the parameters with type, `minimum`, `maximum`, `enum`, `default` and a description in the form `<description> Unit: <unit>. Typical: <typical>. <Build-time|Runtime>.`, and the required parameters. `$id` is `https://github.com/Andste82/sdrk/schema/sdrk-pipeline-v1.json`.

`cargo xtask schema` writes `schema/sdrk-pipeline-v1.json`, `cargo xtask schema --check` compares it with the committed file (CI runs the check). A test in `crates/pipeline/sdrk-pipeline/tests/schema.rs` regenerates the schema in memory and fails if the committed file differs.

**Versioning rule.** Any change of a block description changes the schema. A *compatible* change (a new kind, a new optional parameter, a wider range, a new option, better text) regenerates `sdrk-pipeline-v1.json`. A **breaking** change (a removed parameter, a narrowed range, a changed default, a removed option) must not change the v1 file: it requires a new schema version file `schema/sdrk-pipeline-v2.json` (and a new format version) next to the old one.

## Examples

Three flows in `examples/flows`, all valid against the schema and covered by the tests:

- `fm_receiver.json`: RTL-SDR IQ (`cu8`, 2.4 MHz) to a DDC, an FM demodulator and 48 kHz audio, plus an averaged spectrum frame output fed from the same input buffer.
- `spectrum_only.json`: `cu8` IQ through a DC blocker into a Blackman-Harris spectrum with 8 averages.
- `resample_audio.json`: real `f32` audio at 48 kHz converted to 44.1 kHz by the rational resampler (with `max_taps` raised, because the default of 1023 is too small for this ratio).

## Tests

`crates/pipeline/sdrk-pipeline/tests`: `model` (JSON round trip, canonical form, rates, formats), `schema` (the committed schema, examples against it with `jsonschema`), `validation` (the invalid-flow corpus, all errors at once, limits, cost and memory), `e2e` (the pipeline equals the hand-wired blocks bit for bit, block-size invariance with 10 chunk plans, fan-out, a run-time parameter change), `runner` (drop policies, errors, outputs, `set_param`, `config`, `stats`, `reset`), `no_alloc` (counting allocator) and `property` (random graphs, random bytes and JSON values). The fuzz target `fuzz_targets/pipeline_json.rs` (`cargo +nightly fuzz run pipeline_json` in `crates/pipeline/sdrk-pipeline`) runs in the nightly CI job for five minutes on Linux x86_64 only; fuzzing is not required on Windows and arm64.
