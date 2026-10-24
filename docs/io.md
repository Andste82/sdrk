# File I/O (`sdrk-io`)

The crate `sdrk-io` (layer `crates/io`, std, depends on `sdrk-device` and `serde_json`) reads and writes sample files as streams: readers implement `RxStream`, writers implement `TxStream`. They plug into the pipeline (`push_with_meta`, `pull`) and into the driver conformance checks like a device does ([device.md](device.md#conformance-checks)).

Errors: `sdrk_io::Error` is `Io(String)` (kind `Io`), `Parse { file, reason }` (kind `Parse`), `Unsupported(&'static str)` (kind `Unsupported`), `InvalidArgument(String)` and `Device(sdrk_device::Error)` (kind forwarded).

## Raw files

`RawReader::new(reader, format, rate, start, chunk_samples)` reads any `Read`. Every `read` returns at most `chunk_samples` samples and only whole samples (whole packing groups); a partial sample at the end of the file is dropped; at the end `read` returns zero bytes. The first block carries the anchor given to `with_anchor`. `RawWriter` writes any `Write`, ignores timing and counts the flags of the blocks (`flag_counts`).

## SigMF

`SigmfReader` opens `<base>.sigmf-meta` and `<base>.sigmf-data` (`open` accepts the base path with or without either extension; `new` takes the metadata text and any `Read`). `SigmfWriter::create(base, format, rate)` writes the data file while blocks arrive; `finish()` writes the metadata (`new` and `finish` work on any `Write` and return the text).

| `core:datatype` | Format |
| --- | --- |
| `cu8` | `CU8` |
| `ci8` | `CS8` |
| `ci16_le` | `CS16` |
| `cf32_le` | `CF32` |
| `ri16_le` | `RS16` |
| `rf32_le` | `RF32` |

Every other datatype (big-endian, 32-bit integer, 64-bit float, `ru8`, ...) is `Unsupported`. A format without a datatype cannot be written.

- **Sample rate.** `core:sample_rate` becomes an exact `Rate` when it is an integer or a decimal with at most six fractional digits (`num / 10^k`, reduced: `44100.5` is `88201/2`); anything else is `InvalidArgument`. The writer refuses rates it cannot write exactly.
- **Captures.** The sample at data position `p` of a capture with `core:sample_start = s` and `core:global_index = g` (default `s`) has the stream index `g + (p - s)`; that is `BlockMeta::start`. A block never spans two captures. A capture that starts above the end of the previous one sets `GAP_BEFORE`; one that starts below it is a `Parse` error. A changed `core:frequency` sets `RETUNED`. `core:datetime` becomes `TimeAnchor { index: g, unix_ns, source: Host }` on the first block of the capture. The first capture must start at sample 0.
- **Writer.** A new capture starts at the first block and whenever a block is not contiguous with the previous one (a gap), carries an anchor, carries `RETUNED`, or the frequency set by `set_frequency` changed. `core:global_index` is written only when it differs from `core:sample_start`. `OVERFLOW`, `UNDERRUN` and the burst flags have no place in SigMF and are not stored; `GAP_BEFORE` without a jump of the index cannot be represented.
- **Annotations.** `Annotation { sample_start, sample_count, freq_lower_edge, freq_upper_edge, label, comment }`, written by `annotate` and exposed by `annotations()`. Positions are positions in the data file, as in the SigMF specification, not stream indices; a missing `core:sample_count` reads as 0.
- **Time.** `core:datetime` is `YYYY-MM-DDTHH:MM:SS[.fraction]Z` (UTC). `parse_iso8601_ns` and `format_iso8601_ns` convert to and from unix nanoseconds with their own calendar arithmetic (no date crate); the fraction has one or more digits, digits beyond the ninth are cut off, the writer uses the shortest exact fraction. Years 0001 to 9999 are accepted as long as the value fits an `i64` (1677 to 2262).
- **JSON.** The metadata is written with sorted keys, two-space indentation and LF line ends; the golden vector `io/sigmf_cu8_meta` contains the expected bytes produced by Python's `json`.
- **Not supported.** Versions other than 1.x, `core:dataset` (non-conforming datasets), `core:metadata_only` and `core:header_bytes` other than 0.

## WAV

`WavReader::new(reader, iq, chunk_samples)` reads PCM 16-bit and IEEE float 32-bit, plain or `WAVE_FORMAT_EXTENSIBLE`, one or two channels; unknown chunks before `data` are skipped. A mono file is a real signal (`RS16`, `RF32`); a stereo file must be read with `iq: true` and gives `CS16` or `CF32` (I = left channel); the other combinations are `InvalidArgument`. A data size of `0xFFFFFFFF` (streamed files) reads to the end of the input; a size larger than the data reads what is there.

`WavWriter::new(writer, format, rate)` writes `RS16`, `RF32`, `CS16` or `CF32` with the canonical 44-byte header (the `fmt ` chunk has 16 bytes also for float data) and patches the sizes in `finish()`. The rate must be a whole number of hertz that fits 32 bits; data beyond 4 GiB is `Unsupported`. The golden vectors `io/wav_pcm16_mono` (written with the Python `wave` module) and `io/wav_float32_stereo` (built with `struct`) contain the expected files.

## Command-line examples

`examples/fm-receiver` and `examples/fm-transmitter` use these readers and writers; see [README of the examples](../examples/README.md).
