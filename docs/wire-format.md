# Wire format

The wire format carries samples, frames, events, stream descriptions and parameter changes between the parts of a split pipeline: two threads, two cores, a host and a GPU, or two machines. It is the interface behind the *edges* of the node groups (see [architecture.md](architecture.md#partitioning)). The module `sdrk_pipeline::wire` implements it; the format is **versioned** (version 1) and little-endian only.

A message is a **32-byte header** followed by the payload.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | magic `b"SDRK"` |
| 4 | 1 | version = 1 |
| 5 | 1 | message type: 1 samples, 2 frame, 3 event, 4 hello (stream description), 5 param_set, 6 param_ack |
| 6 | 2 | flags (`BlockFlags` bits) |
| 8 | 2 | stream id |
| 10 | 1 | payload type: 0 raw (format in hello), 1 c32, 2 cq15, 3 f32, 4 q15, 5 UTF-8 JSON |
| 11 | 1 | reserved = 0 |
| 12 | 8 | start index (samples: first sample; frame: `first_sample`) |
| 20 | 4 | count (elements) |
| 24 | 4 | payload length in bytes |
| 28 | 4 | frames: `averaged`; otherwise reserved = 0 |

`BlockFlags` bits: `GAP_BEFORE` 0x01, `OVERFLOW` 0x02, `UNDERRUN` 0x04, `BURST_START` 0x08, `BURST_END` 0x10, `LATE` 0x20, `RETUNED` 0x40; unknown bits are dropped on decode.

## Messages

**Samples (1).** `start` is the index of the first sample, `count` the number of samples, the flags are those of the block. The payload holds the samples: `c32` as interleaved little-endian `f32` I and Q (8 bytes per sample), `cq15` as interleaved `i16` I and Q (4 bytes), `f32` (4 bytes), `q15` as `i16` (2 bytes); raw bytes (payload type 0) are in the format named by the hello of the stream, `count` is then the number of samples (whole packing groups). For the typed payloads `count * element size` must equal the payload length.

**Frame (2).** The payload is `count` little-endian `f32` values (payload type 3), `start` is `FrameMeta.first_sample`, the last header field is `averaged`.

**Hello (4).** Describes a stream; JSON payload, keys in this order:

```json
{"stream":1,"name":"audio","type":"f32","rate_hz":"48000/1","anchor":null}
```

`type` (`c32`, `cq15`, `f32`, `q15`) or, for raw bytes, `format` (a format name such as `cu8`) takes the place of the sample type; `rate_hz` is always `"num/den"`; `anchor` is `null` or `{"index":..,"unix_ns":..,"source":"device"|"host"|"gps"|"unknown"}`. A hello is sent before the first message of a stream; its `stream` field is the stream id of the header.

**Event (3).** JSON payload with the index and the kind, the kind tagged with `type` in `snake_case`; the header `start` repeats the index and `stream` names the stream the event belongs to:

```json
{"index":12345,"kind":{"type":"gap","missing":7}}
```

| `type` | Further keys |
| --- | --- |
| `gap` | `missing` |
| `overflow`, `underrun`, `retuned`, `burst_end`, `late` | none |
| `anchor` | `anchor` (as in the hello) |
| `burst_start` | `tx_time_ns` (number or `null`) |
| `param_applied` | `node` (index), `param` (index in the description), `value` (`{"type":"f64"\|"i64"\|"bool"\|"choice"\|"format","value":..}`), `late` |
| `events_lost` | `count` |

**Param set (5).** A timed parameter change, JSON: `{"node":"tune","name":"frequency_hz","value":610000.0,"at":{"type":"input_index","input":"rf","index":100}}`. `at` is `{"type":"immediate"}`, `{"type":"input_index","input":..,"index":..}` or `{"type":"unix_ns","ns":..}`.

**Param ack (6).** `{"node":"tune","name":"frequency_hz","index":100,"late":false}`; the header `start` repeats `index`.

The JSON of events, hello and parameter messages is produced by serde mirror types of the pipeline (`WireEvent`, `WireAnchor`, `WireParamValue`, `Hello`, `WireParamSet`, `WireParamAck`) with `From` conversions; `sdrk-device` and `sdrk-blocks` stay free of `serde`.

## API

```rust
pub struct Message { pub kind: MessageType, pub flags: BlockFlags, pub stream: u16,
                     pub payload_type: PayloadType, pub start: u64, pub count: u32,
                     pub averaged: u32, pub payload: Vec<u8> }
pub fn encode(msg: &Message, out: &mut Vec<u8>);                 // appends
pub fn decode(buf: &[u8]) -> Result<Decoded>;
pub fn decode_stream(buf: &[u8]) -> Result<(Vec<Message>, usize)>;
pub enum Decoded { Message(Message, usize), NeedMore(usize), Skipped(usize) }
```

`MessageType` and `PayloadType` are `#[non_exhaustive]` enums with exactly the codes above. `Message::samples_c32` (and `_cq15`, `_f32`, `_q15`, `_raw`), `Message::frame`, `Message::hello`, `Message::event`, `Message::param_set` and `Message::param_ack` build messages; `to_data`, `to_frame`, `to_hello`, `to_event`, `to_param_set` and `to_param_ack` read them back.

- `decode` reads the message at the start of the buffer. `NeedMore(n)` says that at least `n` more bytes are needed (the whole message once the header is complete); an empty buffer needs the 32 bytes of a header.
- A wrong magic (also a partial one), a version other than 1, a payload above 64 MiB, an unknown payload type of a known message type, or a payload length that does not fit the element count is a `Wire` error (kind `Parse`).
- **Forward compatibility.** A message of an unknown type is skipped (`Skipped(n)`); `decode_stream` skips them and returns the messages and the number of bytes it consumed. The bytes of an incomplete last message are not consumed, so the caller keeps them and appends the next piece. A later version may add message types, flags and payload types for new messages without breaking a receiver of version 1; a change of the header or of the meaning of existing fields is a new version number.

## Streams and events across a cut

An edge between two groups is a stream. The producer sends a hello, then samples, in order of the index. Events that belong to samples of the stream are sent *before* the samples they refer to; a producer that wants a receiver to turn an event into the flags of a block cuts the samples message at the index of the event, so that the event is at the start of a block. The receiver turns them back into `BlockMeta` for `push_with_meta`: the flags into flags, the anchor into the anchor. The mapping of indices and the delay compensation of anchors are exact node by node, so the split pipeline reports the same events as the single pipeline (`tests/probe_two_threads.rs`). A gap in the samples of a stream arrives as a jump of the start index and is detected by the receiver; its `Gap` event counts the missing samples of the cut stream, not of the original input.

## Examples

The committed files `crates/pipeline/sdrk-pipeline/tests/wire/*.bin` contain one message of each type; the test `the_committed_bytes_are_the_examples` encodes the same messages and compares. They are written by `tests/wire/generate.py`, an independent implementation of the header in Python (`python3 generate.py --check` verifies the files).

Samples, two `c32` samples (1 - 2i, 0.5 + 0.25i), stream 3, start 1000, flag `OVERFLOW` (48 bytes):

```text
5344524b 01 01 0200 0300 01 00 e803000000000000 02000000 10000000 00000000
0000803f 000000c0 0000003f 0000803e
```

Frame, stream 5, `first_sample` 4096, 4 values, 8 averaged (48 bytes):

```text
5344524b 01 02 0000 0500 03 00 0010000000000000 04000000 10000000 08000000
00000000 000028c1 0000a2c1 00004040
```

Event (81 bytes), hello (106), param set (139) and param ack (94) are the 32-byte header followed by the JSON text shown above.
