# Examples

Two command-line programs show the pipeline with real files. They parse their arguments with `std::env::args` only, print the summary to standard error and exit with code 1 and the diagnostics JSON of the pipeline on standard error when the flow is invalid (code 2 for a wrong command line).

## `fm-receiver`

```sh
cargo run -p fm-receiver -- --input rec.sigmf-meta --output audio.wav
cargo run -p fm-receiver -- --input rec.cu8 --format cu8 --rate 2400000 \
    --flow examples/flows/fm_receiver.json --set tune.frequency_hz=600000 --output audio.wav
# stereo: a two-channel WAV file (left, right) at 48 kHz
cargo run -p fm-receiver -- --input rec.cu8 --format cu8 --rate 2400000 \
    --flow examples/flows/fm_stereo_receiver.json --output stereo.wav
```

Reads a SigMF recording, or a raw file with `--format` and `--rate`, pushes it through the flow in `max_chunk` pieces together with the metadata (start index, gaps, anchors, flags) and writes the stream output `audio` as a 32-bit float WAV file (48 kHz for the example flows): one channel for a real (`f32`) stream, two channels for a complex (`C32`) stream, which is stereo audio with the left channel in `re` and the right channel in `im` (the flow `fm_stereo_receiver`; `--set demod.stereo=mono` forces mono output, still in two equal channels). The format and rate of the recording replace those of the flow input. `--set node.param=value` changes a parameter of a node (the value is JSON, or text if it is not JSON; `settings.max_chunk=1024` changes a setting). `--input-name` and `--audio-output` name the input and the output when the flow has several. Without `--flow` the built-in copy of `examples/flows/fm_receiver.json` is used; a file ending in `.yaml` or `.yml` is read as YAML. The events (gaps, overflows, retunes; anchors are only counted) and the per-node statistics go to standard error.

## `fm-transmitter`

```sh
cargo run -p fm-transmitter -- --input audio.wav --output out.cs8
cargo run -p fm-transmitter -- --input audio.wav --output out.sigmf-meta --set mod.deviation_hz=50000
```

Reads a mono WAV file (16-bit PCM or 32-bit float) and pulls the baseband from the raw stream output `rf` of `examples/flows/fm_transmitter.json` (pull mode with a `PullSource` over the WAV reader). The rate of the WAV file replaces the rate of the flow input, as long as the resampler of the flow can convert it (48 kHz and 96 kHz work; 44.1 kHz needs more taps than the resampler allows and is reported as a diagnostic). The output ends with the audio (the `underrun` policy of the output is set to `short`; the tail of the filters is not flushed). A name ending in `.sigmf-meta` or `.sigmf-data` writes a SigMF recording, anything else a raw file.

## Flows

`examples/flows/` has the example flows in JSON with a YAML twin of each; they are described in [docs/pipeline.md](../docs/pipeline.md). `fm_stereo_receiver` is the FM broadcast receiver with stereo decoding (`ddc` to 480 kHz with a 130 kHz cutoff, then `fm_stereo_demod`); `nfm_receiver` is the narrow-band FM receiver (two-way radio, PMR446, amateur 2 m / 70 cm): channel filter, power `squelch`, FM demodulator and the `ctcss` high-pass (`biquad`, order 4, 300 Hz) that removes the CTCSS sub-tone; use `--set tune.frequency_hz=...` to select the channel and `--set gate.threshold_db=...` to move the squelch level. `am_receiver` is the AM broadcast receiver (medium and short wave): channel filter, `agc` with slow time constants (attack 0.1 s, release 0.5 s; a fast AGC would follow the modulation), envelope detector `am_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 4000 Hz, `--set audio_filter.cutoff_hz=...` changes it while the flow runs); `airband_receiver` adds the `squelch` between channel filter and AGC for AM voice in the airband (25 kHz and 8.33 kHz channel spacing are explained in its YAML twin). Both use `--set tune.frequency_hz=...` for the channel and `--set agc.reference_db=...` for the audio level. `ssb_receiver` is the single-sideband receiver (amateur radio, HF utility and maritime): channel filter, `squelch`, `agc` with the SSB defaults, `ssb_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 2800 Hz); the suppressed carrier must be at 0 Hz after the `ddc`, so tune with `--set tune.frequency_hz=...` and choose the sideband with `--set demod.sideband=usb` or `lsb`. `real_input_fm_receiver` receives FM broadcast from a real-sampling device (Airspy R2 or Mini class, `rs16` at 4.8 MS/s): the automatic real `convert`, `real_to_iq` (shift by a quarter of the sample rate and halve the rate: the output centre is the input frequency 1.2 MHz), `ddc` and `fm_demod`; a station at the real frequency `f` is tuned with `--set tune.frequency_hz=...` set to `f - 1.2 MHz` (the example: 1.8 MHz gives +600 kHz).
