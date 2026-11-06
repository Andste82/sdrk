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

`examples/flows/` has the example flows in JSON with a YAML twin of each; they are described in [docs/pipeline.md](../docs/pipeline.md). `fm_transmitter` is the FM broadcast transmitter (`resample` to 240 kHz, the node `shape` = `preemph_limit` with 50 us pre-emphasis and the peak limiter, `fm_mod`, `duc`) and `nfm_transmitter` its narrow-band counterpart for two-way and amateur radio (voice low-pass, 530 us pre-emphasis with a limiter at 0.85, splatter low-pass, 5 kHz deviation, `duc` at +200 kHz); `fm-transmitter --flow examples/flows/nfm_transmitter.json` transmits a WAV file with it, and `nfm_receiver` de-emphasises with 530 us to match. `am_transmitter` (M35) is the AM broadcast transmitter (`biquad` low-pass at 4500 Hz on the audio, `am_mod` with carrier 0.5 and depth 0.8, `duc` by 50 to 2.4 MS/s with the carrier at +600 kHz, `cs8`; `am_receiver` with `--set tune.frequency_hz=600000` receives it) and `ssb_transmitter` the single-sideband voice transmitter (`ssb_mod` with the band 300 to 3000 Hz, `duc` by 50 with the suppressed carrier at 0 Hz, `cs8`; `ssb_receiver` with the same sideband receives it); both are pull-mode flows like `fm_transmitter`, `fm-transmitter --flow examples/flows/am_transmitter.json` (or `ssb_transmitter.json`) transmits a WAV file with either, `--set mod.depth=...` changes the modulation index and `--set mod.sideband=lsb` the sideband. `fm_stereo_transmitter` (M36) is the FM broadcast stereo transmitter for 48 kHz stereo audio as `c32` (left in `re`, right in `im`): `resample` to 192 kHz, `preemph_limit` with one limiter gain for both channels (after the resampler, as in `fm_transmitter`), `fm_stereo_mod` (the multiplex signal with the 19 kHz pilot and the 38 kHz subcarrier; `pilot` can be set to 0 while the flow runs and receivers fall back to mono), `fm_mod` with 75 kHz deviation and `duc` by 10 at +600 kHz to `cs8` at 1.92 MS/s in pull mode; `fm_stereo_demod` decodes it with left on the left. `fm_stereo_receiver` is the FM broadcast receiver with stereo decoding (`ddc` to 480 kHz with a 130 kHz cutoff, then `fm_stereo_demod`); `fm_stereo_receiver_primitives` is the same receiver built from primitives (`quad_demod`, a `biquad` band-pass at 19 kHz, `stereo_pll`, `decimate`, `stereo_matrix`, `deemphasis`): the first flow with a node of two inputs, the edges name the input port with `"port": "a"` or `"port": "b"`; it gives the same audio as `fm_stereo_demod`, bit for bit, from a `cs16` baseband at 480 kHz; `nfm_receiver` is the narrow-band FM receiver (two-way radio, PMR446, amateur 2 m / 70 cm): channel filter, power `squelch`, FM demodulator and the `ctcss` high-pass (`biquad`, order 4, 300 Hz) that removes the CTCSS sub-tone; use `--set tune.frequency_hz=...` to select the channel and `--set gate.threshold_db=...` to move the squelch level. `am_receiver` is the AM broadcast receiver (medium and short wave): channel filter, `agc` with slow time constants (attack 0.1 s, release 0.5 s; a fast AGC would follow the modulation), envelope detector `am_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 4000 Hz, `--set audio_filter.cutoff_hz=...` changes it while the flow runs); `airband_receiver` adds the `squelch` between channel filter and AGC for AM voice in the airband (25 kHz and 8.33 kHz channel spacing are explained in its YAML twin). Both use `--set tune.frequency_hz=...` for the channel and `--set agc.reference_db=...` for the audio level. `ssb_receiver` is the single-sideband receiver (amateur radio, HF utility and maritime): channel filter, `squelch`, `agc` with the SSB defaults, `ssb_demod` and the audio low-pass `audio_filter` (`biquad`, order 4, 2800 Hz); the suppressed carrier must be at 0 Hz after the `ddc`, so tune with `--set tune.frequency_hz=...` and choose the sideband with `--set demod.sideband=usb` or `lsb`. `real_input_fm_receiver` receives FM broadcast from a real-sampling device (Airspy R2 or Mini class, `rs16` at 4.8 MS/s): the automatic real `convert`, `real_to_iq` (shift by a quarter of the sample rate and halve the rate: the output centre is the input frequency 1.2 MHz), `ddc` and `fm_demod`; a station at the real frequency `f` is tuned with `--set tune.frequency_hz=...` set to `f - 1.2 MHz` (the example: 1.8 MHz gives +600 kHz). `hf_am_receiver` (M33) is a short-wave AM receiver for `cs16` IQ at 192 kHz with the noise blanker `noise_blank` before the channel filter and the automatic notch `auto_notch` after the demodulator (`f32` only, audio at 12 kHz); the blanker uses a lookahead of 10 us and no hold (the block defaults, D425), so that its holes in the carrier stay short. Its YAML twin explains when to remove either block: the blanker when there is no impulse noise, the notch for CW and other wanted steady tones. Tune with `--set tune.frequency_hz=...`; `--set nb.threshold_db=...` sets the blanker threshold and `--set notch.mu=...` the step size of the notch (both are also runtime parameters of the pipeline). `gfsk_transmitter` and `gfsk_receiver` (M39) are the two halves of a GFSK packet link at 50 kbit/s with 25 kHz deviation: the transmitter takes packets (`Pipeline::pull` asks a packet source; up to 253 bytes each), frames them with `frame_encode` (4 preamble bytes, PN9 whitening; the gaps between packets are filled with the idle preamble 0x55), modulates with `fsk_mod` (4 samples per bit, BT 0.5) and up-converts with `duc` by 5 to `cs8` at 1 MS/s; the receiver takes `cs16` at 1 MS/s with the channel at 0 Hz through `ddc` (by 5, cutoff 60 kHz), `afc` (up to +-20 kHz; its readings `offset_hz` and `offset_ppm` are in the pipeline statistics, setting its `carrier_hz` to the RF frequency gives the ppm), `fsk_demod` and `frame_decode` (PN9) to a packets output. Both sides must agree on the whitening. `qpsk_transmitter` and `qpsk_receiver` (M40) carry packets of the test framing as QPSK at 25 kBd (50 kbit/s): the transmitter pulls packets from a packet input (`Pipeline::pull` with a `PullSource` that returns `InputData::Packets`; between packets it sends the idle preamble), frames them with `frame_encode`, shapes the symbols with the root-raised-cosine pulse of `psk_mod` (8 samples per symbol), lowers the level by 3 dB so that the pulse peaks fit into `cs8`, and up-converts by 5 to 1 MS/s; the receiver takes `cs16` at 1 MS/s with the signal at 0 Hz, down-converts by 5, normalizes the level with `agc` (`reference_db` 0) and demodulates with `psk_demod` (matched filter, Gardner timing, Costas loop of order 4 with `carrier_bw` 0.04 and `max_offset` 0.1 cycles per symbol, enough for 2.5 kHz of carrier offset) into `frame_decode` with `polarity` both, which accepts frames received with inverted bits. Coherent QPSK can also lock 90 degrees off; then no frame decodes (the YAML twin of the receiver explains it). The test `crates/pipeline/sdrk-pipeline/tests/qpsk_loop.rs` sends 1000 packets through 2 kHz of offset and noise at Eb/N0 10 dB and receives 998 of them. `fm_rds_receiver` (M41) is the FM stereo receiver with RDS for `cs16` IQ at 480 kHz with the station at 0 Hz: `tune` (`ddc` by 2 to 240 kHz, cutoff 100 kHz; `--set tune.frequency_hz=...` for a station off the centre) feeds `stereo` (`fm_stereo_demod` with `audio_decimation` 5, the 48 kHz stereo output `audio`) and `quad` (`quad_demod`, 75 kHz) -> `subcarrier` (`rds_demod`: the 57 kHz subcarrier to soft bits at 1187.5 bit/s) -> `groups` (`rds_decode`: block synchronization, one packet per group) to the packets output `rds`; `fm-receiver` prints PS and RT from it (above).

`aprs_transmitter` and `aprs_receiver` (M39.1) are the two halves of an AX.25/APRS link in Bell 202 AFSK (1200 Bd, mark 1200 Hz, space 2200 Hz) on 48 kHz audio: the transmitter takes AX.25 frames without their FCS as packets (`Pipeline::pull_samples` asks a packet source; 2 to 256 bytes), frames them with `hdlc_encode` (32 opening flags, 4 closing flags, NRZI, flags between frames), modulates 2-FSK with `fsk_mod` at 12 kHz, shifts it to 1700 Hz, takes the real part and resamples to 48 kHz for a sound card or an FM transmitter; the receiver takes that audio (from a sound card or the audio output of an FM receiver flow) through `to_complex`, `rotate` (-1700 Hz), `decimate` (by 4, 1800 Hz), `fsk_demod` and `hdlc_decode` to the packets; `sdrk_proto::ax25::Header::parse(packet.data())` reads the addresses and prints them as `SRC>DST,DIGI1,DIGI2:`, the information field starts at `info_offset()`. `ook_receiver` receives on-off keying at 2000 bit/s from `cu8` at 1 MS/s (`ddc` by 4, `ook_demod`) with the test framing of `frame_decode`; for a real 433 MHz device, replace `frame_decode` by a decoder of its framing and adjust `symbol_rate_hz`. `adsb_receiver` decodes Mode S and ADS-B from an RTL-SDR at 1090 MHz with exactly 2 MS/s (`adsb_decode`); every packet is a frame: with `CRC_OK` the checked frames (DF 17 and 18 as 14 bytes, DF 11 with residue 0 as 7 bytes), with flag bit 16 (`sdrk_blocks::ADSB_ADDRESS_PARITY`) the address/parity frames (DF 0, 4, 5, 11, 16, 20, 21, 24; 7 or 14 bytes) whose address, the CRC-24 of the frame without its last 3 bytes xor those bytes, was seen in a checked frame within the last 60 to 120 s. The downlink format is the first five bits; the ICAO address is bytes 1 to 3 of a checked frame and the syndrome of an address/parity frame (`sdrk_proto::adsb::{df, icao}` and `sdrk_proto::crc::crc24_adsb` compute them):

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
