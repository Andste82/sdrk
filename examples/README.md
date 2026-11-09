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
# stereo with RDS: prints the programme service name and the radiotext
cargo run -p fm-receiver -- --input rds.cs16 --format cs16 --rate 480000 \
    --flow examples/flows/fm_rds_receiver.json --output stereo.wav
```

Reads a SigMF recording, or a raw file with `--format` and `--rate`, pushes it through the flow in `max_chunk` pieces together with the metadata (start index, gaps, anchors, flags) and writes the stream output `audio` as a 32-bit float WAV file (48 kHz for the example flows): one channel for a real (`f32`) stream, two channels for a complex (`C32`) stream, which is stereo audio with the left channel in `re` and the right channel in `im` (the flow `fm_stereo_receiver`; `--set demod.stereo=mono` forces mono output, still in two equal channels). The format and rate of the recording replace those of the flow input. `--set node.param=value` changes a parameter of a node (the value is JSON, or text if it is not JSON; `settings.max_chunk=1024` changes a setting). `--input-name` and `--audio-output` name the input and the output when the flow has several. Without `--flow` the built-in copy of `examples/flows/fm_receiver.json` is used; a file ending in `.yaml` or `.yml` is read as YAML. The events (gaps, overflows, retunes; anchors are only counted) and the per-node statistics go to standard error.

RDS (M41): when the flow has a packets output `rds` (the flow `fm_rds_receiver`; another name with `--rds-output`), `fm-receiver` reads its groups after every push, feeds those with `CRC_OK` to `sdrk_proto::rds::RdsState` and prints `rds: PI 0xD123 PTY 10 PS "SDRK TST"` and `rds: RT "Hello from sdrk!"` to standard error once they are complete and whenever they change; the summary counts the valid and the other groups. A test signal is the reference of M41: `python tests/golden/generate.py --emit rds_reference --emit-out rds.cs16` (8 s of `cs16` at 480 kHz, 30 dB SNR). The same in a program of its own:

```rust
use sdrk_pipeline::{BuildOptions, InputData, Packet, PacketFlags, Registry};
use sdrk_proto::rds::{Group, RdsState};

let flow = std::fs::read_to_string("examples/flows/fm_rds_receiver.json")?;
let mut p = Registry::standard().build_json(&flow, &BuildOptions::default())?;
let mut state = RdsState::new();
let mut groups = [Packet::default(); 16];
for chunk in cs16_bytes.chunks(4 * 4096) {
    p.push("rf", InputData::Raw(chunk))?;
    // (read the stream output "audio" here as well, or it fills up)
    let n = p.read_packets("rds", &mut groups)?;
    for pk in groups.iter().take(n).filter(|pk| pk.flags.contains(PacketFlags::CRC_OK)) {
        if let Some(g) = Group::from_packet(pk) {
            state.update(&g);
        }
    }
    if state.ps_ready() && state.rt_ready() {
        println!("{}: {}", state.ps(), state.rt());
    }
}
```

`ps()` and `rt()` are the ASCII view (bytes outside 0x20 to 0x7E as `?`). `state.ps_text()` and `state.rt_text()` give the same texts with the characters of the RDS character set (umlauts, `€`, `$` at 0xAB): `t.to_string()`, `t.to_utf8(&mut buf)` into caller storage, or `t.with_charset(Charset::Auto)` for stations that send UTF-8 (D690 to D693).

## `fm-transmitter`

```sh
cargo run -p fm-transmitter -- --input audio.wav --output out.cs8
cargo run -p fm-transmitter -- --input audio.wav --output out.sigmf-meta --set mod.deviation_hz=50000
```

Reads a mono WAV file (16-bit PCM or 32-bit float) and pulls the baseband from the raw stream output `rf` of `examples/flows/fm_transmitter.json` (pull mode with a `PullSource` over the WAV reader). The rate of the WAV file replaces the rate of the flow input, as long as the resampler of the flow can convert it (48 kHz and 96 kHz work; 44.1 kHz needs more taps than the resampler allows and is reported as a diagnostic). The output ends with the audio (the `underrun` policy of the output is set to `short`; the tail of the filters is not flushed). A name ending in `.sigmf-meta` or `.sigmf-data` writes a SigMF recording, anything else a raw file.

## Clock drift

When the audio of a flow goes to a sound card (or a transmitter) with its own clock, the queue of the output slowly fills or empties. Set `tunable: true` on the `resample` node and let `PipelineDrift` steer its `ppm` from the fill level of the output (see `examples/flows/resample_audio.yaml` and [docs/blocks.md](../docs/blocks.md), section "Clock drift"):

```rust
use sdrk_blocks::DriftConfig;
use sdrk_pipeline::drift::PipelineDrift;
use sdrk_pipeline::InputData;

// `pipeline` was built from `resample_audio` with `tunable: true` on `to_cd`.
let config = DriftConfig::new(44_100.0, 4_410.0); // 100 ms target fill
let mut drift = PipelineDrift::new(&pipeline, "audio_out", "to_cd", config)?;
// Before every `push` to `audio_in`:
drift.update(&mut pipeline, now_s, sound_card_queued)?;
pipeline.push("audio_in", InputData::F32(&block))?;
// In the audio callback, before every `read_stream` of `audio_out`:
drift.update(&mut pipeline, now_s, sound_card_queued)?;
let n = pipeline.read_stream("audio_out", &mut out)?;
```

Call `update` before every push and before every read: the controller then averages the queue over time. Calling it only after each read lets the fill jump by a whole block whenever the input and output clocks slip by one period, and the controller answers each jump with a spike of several hundred ppm (D314).

## Flows

`examples/flows/` has the example flows in JSON with a YAML twin of each; they are described in [docs/pipeline.md](../docs/pipeline.md). `fm_transmitter` is the FM broadcast transmitter (`resample` to 240 kHz, the node `shape` = `preemph_limit` with 50 us pre-emphasis and the peak limiter, `fm_mod`, `duc`) and `nfm_transmitter` its narrow-band counterpart for two-way and amateur radio (voice low-pass, 530 us pre-emphasis with a limiter at 0.85, splatter low-pass, 5 kHz deviation, `duc` at +200 kHz); `fm-transmitter --flow examples/flows/nfm_transmitter.json` transmits a WAV file with it, and `nfm_receiver` de-emphasises with 530 us to match. `am_transmitter` (M35) is the AM broadcast transmitter (`biquad` low-pass at 4500 Hz on the audio, `am_mod` with carrier 0.5 and depth 0.8, `duc` by 50 to 2.4 MS/s with the carrier at +600 kHz, `cs8`; `am_receiver` with `--set tune.frequency_hz=600000` receives it) and `ssb_transmitter` the single-sideband voice transmitter (`ssb_mod` with the band 300 to 3000 Hz, `duc` by 50 with the suppressed carrier at 0 Hz, `cs8`; `ssb_receiver` with the same sideband receives it); both are pull-mode flows like `fm_transmitter`, `fm-transmitter --flow examples/flows/am_transmitter.json` (or `ssb_transmitter.json`) transmits a WAV file with either, `--set mod.depth=...` changes the modulation index and `--set mod.sideband=lsb` the sideband. `fm_stereo_transmitter` (M36) is the FM broadcast stereo transmitter for 48 kHz stereo audio as `c32` (left in `re`, right in `im`): `resample` to 192 kHz, `preemph_limit` with one limiter gain for both channels (after the resampler, as in `fm_transmitter`), `fm_stereo_mod` (the multiplex signal with the 19 kHz pilot and the 38 kHz subcarrier; `pilot` can be set to 0 while the flow runs and receivers fall back to mono), `fm_mod` with 75 kHz deviation and `duc` by 10 at +600 kHz to `cs8` at 1.92 MS/s in pull mode; `fm_stereo_demod` decodes it with left on the left. `fm_stereo_receiver` is the FM broadcast receiver with stereo decoding (`ddc` to 480 kHz with a 130 kHz cutoff, then `fm_stereo_demod`); `fm_stereo_receiver_primitives` is the same receiver built from primitives (`quad_demod`, a `biquad` band-pass at 19 kHz, `stereo_pll`, `decimate`, `stereo_matrix`, `deemphasis`): the first flow with a node of two inputs, the edges name the input port with `"port": "a"` or `"port": "b"`; it gives the same audio as `fm_stereo_demod`, bit for bit, from a `cs16` baseband at 480 kHz; `nfm_receiver` is the narrow-band FM receiver (two-way radio, PMR446, amateur 2 m / 70 cm): channel filter, power `squelch`, FM demodulator and the `ctcss` high-pass (`biquad`, order 4, 300 Hz) that removes the CTCSS sub-tone; use `--set tune.frequency_hz=...` to select the channel and `--set gate.threshold_db=...` to move the squelch level. `am_receiver` is the AM broadcast receiver (medium and short wave): channel filter, `agc` with slow time constants (attack 0.1 s, release 0.5 s; a fast AGC would follow the modulation), envelope detector `am_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 4000 Hz, `--set audio_filter.cutoff_hz=...` changes it while the flow runs); `airband_receiver` adds the `squelch` between channel filter and AGC for AM voice in the airband (25 kHz and 8.33 kHz channel spacing are explained in its YAML twin). Both use `--set tune.frequency_hz=...` for the channel and `--set agc.reference_db=...` for the audio level. `ssb_receiver` is the single-sideband receiver (amateur radio, HF utility and maritime): channel filter, `squelch`, `agc` with the SSB defaults, `ssb_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 2800 Hz); the suppressed carrier must be at 0 Hz after the `ddc`, so tune with `--set tune.frequency_hz=...` and choose the sideband with `--set demod.sideband=usb` or `lsb`. `real_input_fm_receiver` receives FM broadcast from a real-sampling device (Airspy R2 or Mini class, `rs16` at 4.8 MS/s): the automatic real `convert`, `real_to_iq` (shift by a quarter of the sample rate and halve the rate: the output centre is the input frequency 1.2 MHz), `ddc` and `fm_demod`; a station at the real frequency `f` is tuned with `--set tune.frequency_hz=...` set to `f - 1.2 MHz` (the example: 1.8 MHz gives +600 kHz). `hf_am_receiver` (M33) is a short-wave AM receiver for `cs16` IQ at 192 kHz with the noise blanker `noise_blank` before the channel filter and the automatic notch `auto_notch` after the demodulator (`f32` only, audio at 12 kHz); the blanker uses a lookahead of 10 us and no hold (the block defaults, D425), so that its holes in the carrier stay short. Its YAML twin explains when to remove either block: the blanker when there is no impulse noise, the notch for CW and other wanted steady tones. Tune with `--set tune.frequency_hz=...`; `--set nb.threshold_db=...` sets the blanker threshold and `--set notch.mu=...` the step size of the notch (both are also runtime parameters of the pipeline). `gfsk_transmitter` and `gfsk_receiver` (M39) are the two halves of a GFSK packet link at 50 kbit/s with 25 kHz deviation: the transmitter takes packets (`Pipeline::pull` asks a packet source; up to 253 bytes each), frames them with `frame_encode` (4 preamble bytes, PN9 whitening; the gaps between packets are filled with the idle preamble 0x55), modulates with `fsk_mod` (4 samples per bit, BT 0.5) and up-converts with `duc` by 5 to `cs8` at 1 MS/s; the receiver takes `cs16` at 1 MS/s with the channel at 0 Hz through `ddc` (by 5, cutoff 60 kHz), `afc` (up to +-20 kHz; its readings `offset_hz` and `offset_ppm` are in the pipeline statistics, setting its `carrier_hz` to the RF frequency gives the ppm), `fsk_demod` and `frame_decode` (PN9) to a packets output. Both sides must agree on the whitening. `qpsk_transmitter` and `qpsk_receiver` (M40) carry packets of the test framing as QPSK at 25 kBd (50 kbit/s): the transmitter pulls packets from a packet input (`Pipeline::pull` with a `PullSource` that returns `InputData::Packets`; between packets it sends the idle preamble), frames them with `frame_encode`, shapes the symbols with the root-raised-cosine pulse of `psk_mod` (8 samples per symbol), lowers the level by 3 dB so that the pulse peaks fit into `cs8`, and up-converts by 5 to 1 MS/s; the receiver takes `cs16` at 1 MS/s with the signal at 0 Hz, down-converts by 5, normalizes the level with `agc` (`reference_db` 0) and demodulates with `psk_demod` (matched filter, Gardner timing, Costas loop of order 4 with `carrier_bw` 0.04 and `max_offset` 0.1 cycles per symbol, enough for 2.5 kHz of carrier offset) into `frame_decode` with `rotations` both, which accepts frames received with inverted bits. Coherent QPSK can also lock 90 degrees off; then no frame decodes (the YAML twin of the receiver explains it). The test `crates/pipeline/sdrk-pipeline/tests/qpsk_loop.rs` sends 1000 packets through 2 kHz of offset and noise at Eb/N0 10 dB and receives 998 of them. `fm_rds_receiver` (M41) is the FM stereo receiver with RDS for `cs16` IQ at 480 kHz with the station at 0 Hz: `tune` (`ddc` by 2 to 240 kHz, cutoff 100 kHz; `--set tune.frequency_hz=...` for a station off the centre) feeds `stereo` (`fm_stereo_demod` with `audio_decimation` 5, the 48 kHz stereo output `audio`) and `quad` (`quad_demod`, 75 kHz) -> `subcarrier` (`rds_demod`: the 57 kHz subcarrier to soft bits at 1187.5 bit/s) -> `groups` (`rds_decode`: block synchronization, one packet per group) to the packets output `rds`; `fm-receiver` prints PS and RT from it (above).

`aprs_transmitter` and `aprs_receiver` (M39.1) are the two halves of an AX.25/APRS link in Bell 202 AFSK (1200 Bd, mark 1200 Hz, space 2200 Hz) on 48 kHz audio: the transmitter takes AX.25 frames without their FCS as packets (`Pipeline::pull_samples` asks a packet source; 2 to 256 bytes), frames them with `hdlc_encode` (32 opening flags, 4 closing flags, NRZI, flags between frames), modulates 2-FSK with `fsk_mod` at 12 kHz, shifts it to 1700 Hz, takes the real part and resamples to 48 kHz for a sound card or an FM transmitter; the receiver takes that audio (from a sound card or the audio output of an FM receiver flow) through `to_complex`, `rotate` (-1700 Hz), `decimate` (by 4, 1800 Hz), `fsk_demod` and `hdlc_decode` to the packets; `sdrk_proto::ax25::Header::parse(packet.data())` reads the addresses and prints them as `SRC>DST,DIGI1,DIGI2:`, the information field starts at `info_offset()`. `ook_receiver` receives on-off keying at 2000 bit/s from `cu8` at 1 MS/s (`ddc` by 4 with a 15 kHz channel filter, which keeps a transmitter 10 kHz off in the pass band; widen `cutoff_hz` for devices further off frequency, at the price of sensitivity; `ook_demod`) with the test framing of `frame_decode`; for a real 433 MHz device, replace `frame_decode` by a decoder of its framing and adjust `symbol_rate_hz`. `adsb_receiver` decodes Mode S and ADS-B from an RTL-SDR at 1090 MHz with exactly 2 MS/s (`adsb_decode`); every packet is a frame: with `CRC_OK` the checked frames (DF 17 and 18 as 14 bytes, DF 11 with residue 0 as 7 bytes), with flag bit 16 (`sdrk_blocks::ADSB_ADDRESS_PARITY`) the address/parity frames (DF 0, 4, 5, 11, 16, 20, 21, 24; 7 or 14 bytes) whose address, the CRC-24 of the frame without its last 3 bytes xor those bytes, was seen in a checked frame within the last 60 to 120 s. The downlink format is the first five bits; the ICAO address is bytes 1 to 3 of a checked frame and the syndrome of an address/parity frame (`sdrk_proto::adsb::{df, icao}` and `sdrk_proto::crc::crc24_adsb` compute them):

```rust
use sdrk_pipeline::{InputData, Packet, PacketFlags};

// `pipeline` was built from `examples/flows/adsb_receiver.json`; `bytes` are cu8 samples.
pipeline.push("rf", InputData::Raw(&bytes))?;
let mut packets = [Packet::default(); 64];
let n = pipeline.read_packets("packets", &mut packets)?;
for p in &packets[..n] {
    let d = p.data();
    let df = d[0] >> 3;
    let n = d.len();
    let (icao, kind) = if p.flags == PacketFlags::CRC_OK {
        (u32::from(d[1]) << 16 | u32::from(d[2]) << 8 | u32::from(d[3]), "crc ok")
    } else {
        let parity = u32::from(d[n - 3]) << 16 | u32::from(d[n - 2]) << 8 | u32::from(d[n - 1]);
        (sdrk_proto::crc::crc24_adsb(&d[..n - 3]) ^ parity, "address/parity")
    };
    println!("sample {} df {df} icao {icao:06X} {kind}", p.index);
}
```
`examples/flows/` has the example flows in JSON with a YAML twin of each; they are described in [docs/pipeline.md](../docs/pipeline.md). `fm_transmitter` is the FM broadcast transmitter (`resample` to 240 kHz, the node `shape` = `preemph_limit` with 50 us pre-emphasis and the peak limiter, `fm_mod`, `duc`) and `nfm_transmitter` its narrow-band counterpart for two-way and amateur radio (voice low-pass, 530 us pre-emphasis with a limiter at 0.85, splatter low-pass, 5 kHz deviation, `duc` at +200 kHz); `fm-transmitter --flow examples/flows/nfm_transmitter.json` transmits a WAV file with it, and `nfm_receiver` de-emphasises with 530 us to match. `am_transmitter` (M35) is the AM broadcast transmitter (`biquad` low-pass at 4500 Hz on the audio, `am_mod` with carrier 0.5 and depth 0.8, `duc` by 50 to 2.4 MS/s with the carrier at +600 kHz, `cs8`; `am_receiver` with `--set tune.frequency_hz=600000` receives it) and `ssb_transmitter` the single-sideband voice transmitter (`ssb_mod` with the band 300 to 3000 Hz, `duc` by 50 with the suppressed carrier at 0 Hz, `cs8`; `ssb_receiver` with the same sideband receives it); both are pull-mode flows like `fm_transmitter`, `fm-transmitter --flow examples/flows/am_transmitter.json` (or `ssb_transmitter.json`) transmits a WAV file with either, `--set mod.depth=...` changes the modulation index and `--set mod.sideband=lsb` the sideband. `fm_stereo_transmitter` (M36) is the FM broadcast stereo transmitter for 48 kHz stereo audio as `c32` (left in `re`, right in `im`): `resample` to 192 kHz, `preemph_limit` with one limiter gain for both channels (after the resampler, as in `fm_transmitter`), `fm_stereo_mod` (the multiplex signal with the 19 kHz pilot and the 38 kHz subcarrier; `pilot` can be set to 0 while the flow runs and receivers fall back to mono), `fm_mod` with 75 kHz deviation and `duc` by 10 at +600 kHz to `cs8` at 1.92 MS/s in pull mode; `fm_stereo_demod` decodes it with left on the left. `fm_stereo_receiver` is the FM broadcast receiver with stereo decoding (`ddc` to 480 kHz with a 130 kHz cutoff, then `fm_stereo_demod`); `fm_stereo_receiver_primitives` is the same receiver built from primitives (`quad_demod`, a `biquad` band-pass at 19 kHz, `stereo_pll`, `decimate`, `stereo_matrix`, `deemphasis`): the first flow with a node of two inputs, the edges name the input port with `"port": "a"` or `"port": "b"`; it gives the same audio as `fm_stereo_demod`, bit for bit, from a `cs16` baseband at 480 kHz; `nfm_receiver` is the narrow-band FM receiver (two-way radio, PMR446, amateur 2 m / 70 cm): channel filter, power `squelch`, FM demodulator and the `ctcss` high-pass (`biquad`, order 4, 300 Hz) that removes the CTCSS sub-tone; use `--set tune.frequency_hz=...` to select the channel and `--set gate.threshold_db=...` to move the squelch level. `am_receiver` is the AM broadcast receiver (medium and short wave): channel filter, `agc` with slow time constants (attack 0.1 s, release 0.5 s; a fast AGC would follow the modulation), envelope detector `am_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 4000 Hz, `--set audio_filter.cutoff_hz=...` changes it while the flow runs); `airband_receiver` adds the `squelch` between channel filter and AGC for AM voice in the airband (25 kHz and 8.33 kHz channel spacing are explained in its YAML twin). Both use `--set tune.frequency_hz=...` for the channel and `--set agc.reference_db=...` for the audio level. `ssb_receiver` is the single-sideband receiver (amateur radio, HF utility and maritime): channel filter, `squelch`, `agc` with the SSB defaults, `ssb_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 2800 Hz); the suppressed carrier must be at 0 Hz after the `ddc`, so tune with `--set tune.frequency_hz=...` and choose the sideband with `--set demod.sideband=usb` or `lsb`. `real_input_fm_receiver` receives FM broadcast from a real-sampling device (Airspy R2 or Mini class, `rs16` at 4.8 MS/s): the automatic real `convert`, `real_to_iq` (shift by a quarter of the sample rate and halve the rate: the output centre is the input frequency 1.2 MHz), `ddc` and `fm_demod`; a station at the real frequency `f` is tuned with `--set tune.frequency_hz=...` set to `f - 1.2 MHz` (the example: 1.8 MHz gives +600 kHz). `hf_am_receiver` (M33) is a short-wave AM receiver for `cs16` IQ at 192 kHz with the noise blanker `noise_blank` before the channel filter and the automatic notch `auto_notch` after the demodulator (`f32` only, audio at 12 kHz); the blanker uses a lookahead of 10 us and no hold (the block defaults, D425), so that its holes in the carrier stay short. Its YAML twin explains when to remove either block: the blanker when there is no impulse noise, the notch for CW and other wanted steady tones. Tune with `--set tune.frequency_hz=...`; `--set nb.threshold_db=...` sets the blanker threshold and `--set notch.mu=...` the step size of the notch (both are also runtime parameters of the pipeline). `gfsk_transmitter` and `gfsk_receiver` (M39) are the two halves of a GFSK packet link at 50 kbit/s with 25 kHz deviation: the transmitter takes packets (`Pipeline::pull` asks a packet source; up to 253 bytes each), frames them with `frame_encode` (4 preamble bytes, PN9 whitening; the gaps between packets are filled with the idle preamble 0x55), modulates with `fsk_mod` (4 samples per bit, BT 0.5) and up-converts with `duc` by 5 to `cs8` at 1 MS/s; the receiver takes `cs16` at 1 MS/s with the channel at 0 Hz through `ddc` (by 5, cutoff 60 kHz), `afc` (up to +-20 kHz; its readings `offset_hz` and `offset_ppm` are in the pipeline statistics, setting its `carrier_hz` to the RF frequency gives the ppm), `fsk_demod` and `frame_decode` (PN9) to a packets output. Both sides must agree on the whitening. `qpsk_transmitter` and `qpsk_receiver` (M40) carry packets of the test framing as QPSK at 25 kBd (50 kbit/s): the transmitter pulls packets from a packet input (`Pipeline::pull` with a `PullSource` that returns `InputData::Packets`; between packets it sends the idle preamble), frames them with `frame_encode`, shapes the symbols with the root-raised-cosine pulse of `psk_mod` (8 samples per symbol), lowers the level by 3 dB so that the pulse peaks fit into `cs8`, and up-converts by 5 to 1 MS/s; the receiver takes `cs16` at 1 MS/s with the signal at 0 Hz, down-converts by 5, normalizes the level with `agc` (`reference_db` 0) and demodulates with `psk_demod` (matched filter, Gardner timing, Costas loop of order 4 with `carrier_bw` 0.04 and `max_offset` 0.1 cycles per symbol, enough for 2.5 kHz of carrier offset) into `frame_decode` with `rotations` both, which accepts frames received with inverted bits. Coherent QPSK can also lock 90 degrees off; then no frame decodes (the YAML twin of the receiver explains it). The test `crates/pipeline/sdrk-pipeline/tests/qpsk_loop.rs` sends 1000 packets through 2 kHz of offset and noise at Eb/N0 10 dB and receives 998 of them. `fm_rds_receiver` (M41) is the FM stereo receiver with RDS for `cs16` IQ at 480 kHz with the station at 0 Hz: `tune` (`ddc` by 2 to 240 kHz, cutoff 100 kHz; `--set tune.frequency_hz=...` for a station off the centre) feeds `stereo` (`fm_stereo_demod` with `audio_decimation` 5, the 48 kHz stereo output `audio`) and `quad` (`quad_demod`, 75 kHz) -> `subcarrier` (`rds_demod`: the 57 kHz subcarrier to soft bits at 1187.5 bit/s) -> `groups` (`rds_decode`: block synchronization, one packet per group) to the packets output `rds`; `fm-receiver` prints PS and RT from it (above). `fm_stereo_rds_transmitter` (M42) is `fm_stereo_transmitter` with RDS: the limited 192 kHz audio clocks `rds_encode` (PI 0xD123, PTY 10), whose shaped RDS baseband goes to port `b` of `fm_stereo_rds_mod` (`main` 0.87, `pilot` 0.09, `rds` 0.04 on 57 kHz, three times the pilot phase); the station name and the radiotext arrive as control packets on the input `text` (`Pipeline::push_packets`; `0x01` + 8 bytes the name, `0x02` + up to 64 bytes the radiotext, `0x03` + 8 bytes one raw group, the blocks A, B, C, D big-endian), and in pull mode the source is asked for the audio only, so push the packets between two `pull` calls. The receiver chain of `fm_rds_receiver` decodes name, radiotext and raw groups from it (the loop test in `crates/pipeline/sdrk-pipeline/tests/rds_tx.rs`).

`cw_transmitter`, `cw_receiver`, `cw_tone_transmitter` and `cw_tone_receiver` (M47) send and receive Morse code (CW). `cw_transmitter` takes text as packets (up to 256 bytes each; letters, digits, `. , : ? ' - / ( ) " = + @`, a space is a word gap, `+ = (` are the prosigns AR BT KN and the byte `0x04` is SK) at the rate of its input (8000 Hz, the unit at 20 words per minute is 480 samples) and makes a carrier 10 kHz above the tuner centre, 240 kS/s `cs8`, in pull mode (the carrier is off when no text waits); every packet ends with a word gap. `cw_receiver` takes `cu8` at 240 kS/s (an RTL-SDR with `--rate 240000`; `--set tune.frequency_hz=...` for a carrier elsewhere in the 120 kHz) and returns one packet per word. The two tone flows work on 8 kHz audio: `cw_tone_transmitter` makes a keyed 700 Hz tone for the microphone input of an SSB transmitter or a sound card, `cw_tone_receiver` takes the audio of an SSB receiver (`--set tone.cutoff_hz=...` sets the pitch of the band-pass). The decoder finds the speed (5 to 40 words per minute) itself: start with a preamble such as `VVV` and give it about a unit of silence or noise first, because the first block after the start only learns the noise level. `crates/pipeline/sdrk-pipeline/tests/morse_flows.rs` closes the loops.

```rust
use sdrk_pipeline::{InputData, Packet, PacketFlags, PullError, PullSource};

// Receive: `pipeline` was built from `examples/flows/cw_receiver.json`; `bytes` are cu8 samples.
pipeline.push("rf", InputData::Raw(&bytes))?;
let mut words = [Packet::default(); 64];
let n = pipeline.read_packets("words", &mut words)?;
for w in &words[..n] {
    println!("{} (sample {})", String::from_utf8_lossy(w.data()), w.index);
}

// Send: a packet source for `Pipeline::pull` on `cw_transmitter` (cs8 bytes); an empty answer
// means "nothing to send now", and the carrier stays off.
struct Text {
    packets: Vec<Packet>,
    next: usize,
}
impl PullSource for Text {
    fn provide(&mut self, _input: &str, _max: usize) -> Result<InputData<'_>, PullError> {
        let from = self.next;
        self.next = (from + 1).min(self.packets.len());
        Ok(InputData::Packets(&self.packets[from..self.next]))
    }
}
let packets = vec![Packet::new(0, PacketFlags::NONE, b"VVV CQ DE DL1ABC K")?];
let mut source = Text { packets, next: 0 };
let mut rf = vec![0u8; 16384];
let (n, _) = pipeline.pull("rf", &mut rf, &mut source)?;
```

`qam16_transmitter` and `qam16_receiver` (M44) carry packets of the test framing as 16-QAM at 25 kBd (100 kbit/s, four bits per symbol). The transmitter works as `qpsk_transmitter` (pull mode, a `PullSource` that returns `InputData::Packets`, `frame_encode` with 4 preamble bytes) with `qam_mod` (Gray mapping, root-raised-cosine pulse with roll-off 0.35, 8 samples per symbol), `scale` -6 dB so that the pulse peaks (about 1.5 at unit average power) fit into `cs16`, and `duc` by 5 to 1 MS/s. Keep its packet source busy: between packets the encoder sends the preamble bits, which in 16-QAM are one constant symbol, and the receiver's timing loop and blind equalizer cannot train on it (send filler packets that the application ignores). The receiver takes `cs16` at 1 MS/s with the signal at 0 Hz: `ddc` by 5, `agc` (`reference_db` 0, attack and release 20 ms), `qam_demod` (matched filter, Gardner timing with `sync_bw` 0.002, an equalizer of 15 taps, the decision-directed carrier loop with `carrier_bw` 0.1, wide enough for a carrier 1 kHz off) and `frame_decode` with `rotations` four and `bits_per_symbol` 4, which finds the sync word in all four 90 degree rotations of the square constellation and reports the rotation in bits 16 and 17 of the packet flags (`sdrk_proto::frame::rotation_of`); test packets for `CRC_OK` with `flags.contains(PacketFlags::CRC_OK)`, not with `==`. For a carrier known to within about 25 Hz, `carrier_bw` 0.005 (the default) on the node `demod` gives less phase jitter. The test `crates/pipeline/sdrk-pipeline/tests/qam_loop.rs` sends 1000 packets through mild multipath, 1 kHz of offset and noise at Eb/N0 16 dB and receives 992 of them at every rotation. `order` 64 on both sides gives 64-QAM (`rate_hz` 150000 for 25 kBd, `bits_per_symbol` 6), which pulls in only small carrier offsets (`docs/open-items.md`, I44-3).

`pmr446_monitor` (M45) listens to eight PMR446 channels at once with one polyphase channelizer. The input is an RTL-SDR recording (`cu8` at 2.4 MS/s) centred on PMR446 channel 9 (446.10625 MHz), so the channel `n` (446.00625 + 0.0125 (n - 1) MHz) is the signed lane `n - 9` of the channelizer. `dec` (`ddc` by 12) gives the 200 kS/s band of 16 channels, `chan` (`channelize`, 16 channels, 24 taps per branch, flat to 0.4 of the spacing and 60 dB down from 0.6) gives one lane stream of 16 channels at 25 kS/s, and eight chains `selN` (`channel_select`, channel `N - 9`) -> `gateN` (`squelch`) -> `demodN` (`fm_demod`, 2.5 kHz deviation, 530 us de-emphasis) for the channels 1 to 8 feed a tree of seven `add` nodes; `out` (`resample` to 48 kHz with the speech band as its low-pass) writes the stream `audio`:

```sh
rtl_sdr -f 446106250 -s 2400000 pmr.cu8
cargo run -p fm-receiver -- --input pmr.cu8 --format cu8 --rate 2400000 \
    --flow examples/flows/pmr446_monitor.json --output pmr.wav
```

The sum is not normalized: with the squelches, usually one channel is open at a time; for several open channels put a `scale` between `mix7` and `out` (the YAML comment shows where). The YAML twin also explains how to add channels 9 to 16 (another eight chains on the lanes 0 to 7). `crates/pipeline/sdrk-pipeline/tests/pmr446.rs` runs the flow on NFM channel scenes of the testkit (`sdrk_testkit::nfm::ChannelScene`).

`ism433_pulses` and `ism868_pulses` (M45.1) receive every short-range device of an ISM band at once, without knowing its frequency: weather stations, thermometers, door contacts and remote controls, whose crystals or SAW resonators are tens of kHz off. `ism433_pulses` takes an RTL-SDR recording (`cu8` at 2 MS/s) centred at 433.92 MHz, which holds the whole band 433.05 to 434.79 MHz; `chan` (`channelize`, 16 channels of 125 kHz) splits it into overlapping channels, and `det` (`band_pulse_detect`) runs a pulse detector after the method of `rtl_433` in every channel (OOK and two-level FSK), merges what neighbouring channels see of the same transmission and writes one pulse train per transmission to the packets output `pulses`. `ism868_pulses` does the same at 868.5 MHz with 8 channels of 250 kHz for the wider FSK sensors (868.3 MHz and its neighbours and 868.95 MHz; devices above 869.0 MHz need the centre moved up to 868.95 MHz, see its YAML twin). Each packet is a train of `sdrk_proto::pulses` (modulation, signed channel, frequency offset from the centre, level, noise, the pulse and gap widths in steps of the channel rate, 250 or 500 kHz); a train of more than 30 pulses comes as several packets flagged `CONTINUED`, which `pulses::Assembler` joins, and `pulses::write_json` writes the JSON form with the absolute frequency (the RF centre it adds is the build parameter `center_hz` of `det`, which a tool reads from the flow description; the packets carry the offset from the centre). An FSK transmission comes as one train whose transitions are taken from the frequency discriminator and fitted to the grid of its symbol clock. A wideband event (an impulse or a burst of interference over the whole band) gives one train in both flows, not one per channel. Single-pulse trains are passed on; the decoders reject garbage by length, checksum and repeats. The device protocols (M46) decode the trains; `--set det.min_snr_db=...` (runtime) moves the trigger level.

```sh
rtl_sdr -f 433920000 -s 2000000 ism.cu8
```

```rust
use sdrk_pipeline::{InputData, Packet, PipelineSpec};
use sdrk_proto::pulses::{self, Assembler, Pair};

// `pipeline` was built from `flow`, the text of `examples/flows/ism433_pulses.json`; `bytes`
// are cu8 samples. The RF centre of the absolute frequency is the flow's `center_hz` of `det`.
let center_hz = PipelineSpec::from_json(&flow)?.nodes["det"].params["center_hz"]
    .as_i64()
    .unwrap_or(0);
let mut trains = Assembler::new(vec![Pair::default(); 1200]);
pipeline.push("rf", InputData::Raw(&bytes))?;
let mut packets = [Packet::default(); 256];
let n = pipeline.read_packets("pulses", &mut packets)?;
for p in &packets[..n] {
    if let Some(t) = trains.push(p)? {
        let mut line = String::new();
        pulses::write_json(&mut line, &t.header, t.pairs.iter().copied(), center_hz)?;
        println!("{line}"); // {"mod":"ook","freq_hz":433947312,"rssi_db":-21.4,...,"pulses":[[50,100],...]}
    }
}
```

The packets can also be stored as JSON Lines with `sdrk_io::packets::write_jsonl` (hex payload) and read back with `read_jsonl`. `crates/pipeline/sdrk-pipeline/tests/ism_flows.rs` runs both flows on `cu8` scenes of `sdrk_testkit::ism`.
