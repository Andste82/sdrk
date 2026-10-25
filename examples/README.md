# Examples

Two command-line programs show the pipeline with real files. They parse their arguments with `std::env::args` only, print the summary to standard error and exit with code 1 and the diagnostics JSON of the pipeline on standard error when the flow is invalid (code 2 for a wrong command line).

## `fm-receiver`

```sh
cargo run -p fm-receiver -- --input rec.sigmf-meta --output audio.wav
cargo run -p fm-receiver -- --input rec.cu8 --format cu8 --rate 2400000 \
    --flow examples/flows/fm_receiver.json --set tune.frequency_hz=600000 --output audio.wav
```

Reads a SigMF recording, or a raw file with `--format` and `--rate`, pushes it through the flow in `max_chunk` pieces together with the metadata (start index, gaps, anchors, flags) and writes the stream output `audio` as a mono 32-bit float WAV file (48 kHz for the example flow). The format and rate of the recording replace those of the flow input. `--set node.param=value` changes a parameter of a node (the value is JSON, or text if it is not JSON; `settings.max_chunk=1024` changes a setting). `--input-name` and `--audio-output` name the input and the output when the flow has several. Without `--flow` the built-in copy of `examples/flows/fm_receiver.json` is used; a file ending in `.yaml` or `.yml` is read as YAML. The events (gaps, overflows, retunes; anchors are only counted) and the per-node statistics go to standard error.

## `fm-transmitter`

```sh
cargo run -p fm-transmitter -- --input audio.wav --output out.cs8
cargo run -p fm-transmitter -- --input audio.wav --output out.sigmf-meta --set mod.deviation_hz=50000
```

Reads a mono WAV file (16-bit PCM or 32-bit float) and pulls the baseband from the raw stream output `rf` of `examples/flows/fm_transmitter.json` (pull mode with a `PullSource` over the WAV reader). The rate of the WAV file replaces the rate of the flow input, as long as the resampler of the flow can convert it (48 kHz and 96 kHz work; 44.1 kHz needs more taps than the resampler allows and is reported as a diagnostic). The output ends with the audio (the `underrun` policy of the output is set to `short`; the tail of the filters is not flushed). A name ending in `.sigmf-meta` or `.sigmf-data` writes a SigMF recording, anything else a raw file.

## Flows

`examples/flows/` has the example flows in JSON with a YAML twin of each; they are described in [docs/pipeline.md](../docs/pipeline.md). `nfm_receiver` is the narrow-band FM receiver (two-way radio, PMR446, amateur 2 m / 70 cm): channel filter, power `squelch`, FM demodulator; use `--set tune.frequency_hz=...` to select the channel and `--set gate.threshold_db=...` to move the squelch level.
