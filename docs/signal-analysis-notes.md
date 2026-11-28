# Signal analysis with SDRk: notes from the first case

Status: draft notes, 2026-11-28. Written after identifying an unknown signal in a recording; the
second half proposes `signal_analysis` blocks. Nothing here is decided.

## 1. The case

`baseband_427977000Hz_13-19-13_28-11-2026.wav`: 16-bit stereo WAV, I in the left and Q in the right
channel, 1 024 000 S/s, 5.857 s, centre 427.977 MHz (from the file name). It was recorded with an
8-bit receiver (RTL-SDR class) and a foreign signal showed up by chance. Question: what is it?

A prototype that performs every step below with SDRk blocks (a small Rust program outside the
repository, `sdrk-analyze`) was used; its functions are named where they matter.

## 2. Procedure, in the order it was done

Each step: what it answers, how it was done, what it gave for this recording.

### 2.1 Look at the file before any DSP

- Header: format, channels, bits, rate, length. Here 16 bit, 2 channels, 1 024 000 S/s.
- Level: RMS -44 dBFS, peak 0.024 of full scale, no clipping, DC about 0.
- Quantisation: 28 (I) and 32 (Q) distinct values, in clusters 256 apart, so the source is 8 bit scaled to 16
  and only about +-2 steps are used. The ADC is at its noise floor; the gain was far too low.
  This limits everything below (plateau SNR 9 to 12 dB).

### 2.2 Is there anything in the total power?

Power per 1 ms frame, histogram. Here: -44 dB +-0.7 dB, no bursts. **This looked like "no
signal" and was the wrong conclusion**: the signals are continuous and narrow; the power of the
whole 1 MHz is dominated by noise. A wideband power test is blind to a 20 kHz signal that is
10 dB above the noise in its own band (the same reason `BurstDetector` in `sdrk-qam` watches a
narrow band).

### 2.3 Overview spectrum, find the signals

- Block `spectrum`: `fft_size` 16384, `averaging` 8, Hann. 45 frames of 62.5 Hz bins.
- Average the frames in linear power, convert back to dB.
- Floor = median of the spectrum. Smooth over 2 kHz. A signal is a region more than 3 dB above the
  floor and wider than 5 kHz. Centre and width from the -3 dB crossings of the region.
- Result: three signals at -113.0, -63.3 and +111.1 kHz from the centre, 11 to 14 dB over the floor.

### 2.4 Select a channel

Block `ddc`: `frequency_hz` = the centre found, `decimation` 16 (1.024 MS/s to 64 kS/s), `cutoff_hz`
27000, `transition_hz` 5000. All later steps work on this channel.

### 2.5 Channel spectrum: shape and bandwidth

Block `spectrum` on the channel (`fft_size` 2048, `averaging` 16). From the averaged spectrum:
plateau level (median within +-6 kHz), noise (median at +-25 kHz), width at -3/-6/-10/-20 dB
relative to the plateau, 99 % occupied bandwidth, spectral centroid, plateau ripple.
Result: plateau 9 to 12 dB over noise, -3 dB width 14.5 to 18.7 kHz, -6 dB width about 21 kHz,
centroid +332, +562, +1002 Hz. The -10/-20 dB and 99 % figures are useless at this SNR (they run
into the noise and the channel filter): a measurement must say when the SNR is too low.

### 2.6 Symbol rate from the spectrum of the envelope

Blocks `mag`, `to_complex`, `spectrum` (`fft_size` 4096, `averaging` 8, 15.6 Hz bins). A linearly
modulated signal with a pulse filter has a cyclostationary envelope: a spectral line at the symbol
rate. Search for lines that stand more than 6 dB above the local median (+-40 bins), ignoring DC.
Result: **18000.0 Hz** in all three channels, +16 to +20 dB, the next lines 10 dB lower. This is
the most informative single step: one line, Hz precision, no knowledge of the modulation needed.

### 2.7 Time structure and envelope statistics

From the channel samples: power per ms (mean, std, min, max), peak/RMS, coefficient of variation
and excess kurtosis of |x|^2, autocorrelation of the power in 0.5 ms steps up to 300 ms.
Result: std of the power 0.35 to 0.6 dB per ms, so a continuous transmission; no slot or frame
period found (the best lags of the power autocorrelation have almost equal values, 0.09 to 0.49, no
distinct peak; the lags of 14.17 and 56.67 ms were not looked at specifically). Peak/RMS 7 to 8.6 dB and kurtosis are
dominated by the noise at this SNR and did not discriminate.

### 2.8 Matched filter and symbol timing

`resample` to **72 000 S/s** (4 samples per symbol), `pulse_match` (`symbol_rate_hz` 18000,
`rolloff` 0.35, `span` 10), `symbol_sync` (`symbol_rate_hz` 18000, Gardner, `loop_bw` 0.005).
`pulse_match` needs the sample rate to be within 1 % of an integer multiple (2 to 1024) of the
symbol rate; 64 kS/s / 18 kBd = 3.56 is not, hence the `resample`. The roll-off 0.35 was assumed,
not measured (see 3.4). Output: 103 435 symbols per channel.

### 2.9 Which modulation: powers of the symbols

Normalise to unit RMS; for p = 1, 2, 4, 8 take s^p over all symbols and the power spectrum (Hann,
2^16 points) and note the peak over the median and its place in cycles per symbol:

- BPSK has a line at p = 2, QPSK at p = 4, 8PSK and pi/4-QPSK at p = 8.
- Strong lines appeared at several p (26 to 37 dB), more than noise explains; the stable narrow
  lines seen in the first overview (17 dB over the noise, 15.6 Hz bins, constant over 5.9 s) are
  deterministic content in the signal, and they contaminate every power. The line of s^8 was
  18 to 24 dB over the median in all three, at the same offset as the line of s^4 for channel 2.
- The offset from s^8 is known only **modulo 1/8 cycle per symbol (2250 Hz)**.

### 2.10 Resolve the ambiguity, then look at the phases

The ambiguity matters: derotating a QPSK signal by a wrong multiple of 45 degrees per symbol makes
it look like pi/4-QPSK (successive symbols alternate between two rotated sets). Resolution: take
the candidate nearest to the centroid of the channel spectrum (2.5). The candidates are 2250 Hz
apart and the centroid is good to a few hundred Hz, so it is unique here. Offsets: +349, +628,
+1275 Hz from the `ddc` centres.

Then, with the symbols derotated: the mean of exp(j 4 phi) of the **even and the odd symbols is
180 degrees apart** in all three channels (+132/-48, +16/-164, +137/-43 degrees, magnitude 0.16 to
0.18), and the phase histograms (16 bins) have four peaks per turn that move by 45 degrees between
even and odd symbols. That is the signature of **pi/4-QPSK** (differentially encoded in practice).

### 2.11 Carrier frequencies, spacing and raster

RF carrier = record centre + `ddc` centre + offset: 427.86435, 427.91435 and 428.08934 MHz.
The spacing is 49.998 and 174.990 kHz (2 and 7 channels of 25 kHz). All three are 14.34 kHz
(+-6 Hz) above a multiple of 25 kHz. If the network raster is offset by 12.5 kHz, the receiver
reads 1.84 kHz (4.3 ppm) high; with other raster offsets the error would be 33 ppm (offset 0),
19 ppm (6.25 kHz) or -10 ppm (18.75 kHz). The raster of the network decides; the data cannot.

### 2.12 Conclusion for this recording

Three continuous carriers, 25 kHz grid, symbol rate 18.000 kBd, root-raised-cosine like pulse
(0.35 consistent), pi/4-QPSK: **consistent with TETRA** (18 kBd, pi/4-DQPSK, 25 kHz channels,
36 kbit/s). Section 6 then found the burst structure of TETRA in the symbols (frequency correction
field and synchronization training sequence), which settles it.

## 3. Pitfalls met on the way

1. **Wideband power hides narrow signals** (2.2). Detect in band.
2. **Coarse quantisation is not circular noise**: E[x^4] is not zero, so a fourth-power line shows
   up at 0 Hz in 8-bit data (the same artefact made `sdrk-qam` see a carrier in random bytes).
   Remove the mean per block, and do not read a line at 0 Hz of s^4 as a modulation.
3. **Pulse filters need an integer number of samples per symbol** (`rate_unsupported`), so the
   rate has to be brought to an integer multiple first (`resample`).
4. **Offset from s^p is ambiguous modulo 1/p cycle per symbol**, and for p = 8 that is the
   difference between QPSK and pi/4-QPSK (2.10). A second, independent coarse estimate (the
   centroid) is needed.
5. **-3 dB widths measured from the highest point of a spectrum** are biased when narrow lines sit on
   the plateau (the overview widths of 14 to 16 kHz are lower than the channel widths relative to the
   plateau median). Use the plateau median.
6. **SNR limits what can be said.** At 9 to 12 dB plateau SNR the 99 % bandwidth and the -20 dB
   width are noise. Every figure should come with a validity flag.
7. A flow needs a `name`; `read_frame` has to be called after every `push` or a `drop_oldest` queue
   loses frames.

## 4. What was missing: proposal for `signal_analysis` blocks

Every step above except 2.4, 2.8 and the filters was written as Rust next to the blocks
(the prototype `sdrk-analyze`, outside the repository). They are the same few measurements for every unknown
signal, so they would be useful as blocks (or as one analysis API) that return the figures
directly. A first list, ordered by how much each would have saved:

| Name | Input | Output (figures) | Notes |
| --- | --- | --- | --- |
| `signal_find` | spectrum frames, or `C32` | list of signals: centre, -3 dB width, plateau level, SNR | steps 2.3 and 2.5 in one; median floor, smoothing, hysteresis; returns "none" |
| `occupied_bandwidth` | channel `C32` | -3/-6/-10/-20 dB widths, 99 % width, centroid, ripple, validity | with a validity flag (SNR) |
| `symbol_rate_estimate` | channel `C32` | symbol rate (Hz), line strength, runner-up lines | envelope-spectrum lines (2.6); also |x|^2 and a cyclic-autocorrelation variant; the cheapest and most useful block |
| `rolloff_estimate` | channel spectrum + symbol rate | roll-off (0 to 1) | fit of a raised-cosine / RRC edge; validates the assumption in 2.8 |
| `power_stats` | channel `C32` | power per interval (mean, std, min, max), duty cycle, PAPR, |x|^2 CV and kurtosis, period (autocorrelation peak) | burst/continuous, slot and frame period of TDMA |
| `modulation_estimate` | channel `C32` + symbol rate | class and scores: AM, FM, 2-/4-FSK (deviation, index), OOK, BPSK, QPSK, pi/4-QPSK, 8PSK, 16/64-QAM, multicarrier | features: s^p lines with the modulo ambiguity resolved against the centroid, even/odd alternation, differential phase histogram, cumulants C40/C42, FM discriminator histogram; runs the matched filter and timing itself |
| `carrier_estimate` | channel `C32` | offset (Hz), precision, ambiguity resolved flag | s^p line + centroid, as 2.10 |
| `raster_check` | list of carriers | spacing, common remainder, ppm for a given raster | host-side helper, not a block |

Blocks of this kind produce text or a small struct, not a stream. There is a precedent for
reports as packets: `adsb_parse` emits one JSON text packet per message.

### 4.1 Questions for the maintainer

1. One block per figure, or one `signal_analysis` composite with several outputs (like
   `qam_frame_decode` is a composite with a map of parameters)? The composite would run the channel
   chain once.
2. Output type: JSON text packets (precedent: `adsb_parse`), a frame of `f32` with a fixed layout,
   or a Rust API on a block that the tool reads (`acquired_frequency()` of `qam_demod` is such a
   getter)?
3. Targets: the figures need FFTs of 2^12 to 2^16 points and many seconds of samples. That is a
   host (`std`, `f32`) feature; the embedded targets are on hold anyway. State that in the block
   description so the MCU rule ("check the algorithm against the SDRk goals") is answered.
4. Test: synthetic signals from `sdrk-testkit` with known symbol rate, roll-off, constellation,
   offset and SNR (the tests would assert recovery within a tolerance, and the validity flag at low
   SNR), plus this recording as a regression case if the file may be added.
5. SIMD: the FFT and the powers of the symbols are SIMD-able through the existing `fft` and `mul`
   kernels; the line search and the statistics are not hot.

## 5. Not done yet for this signal

- Decode the BSCH (SYNC PDU) and SYSINFO: colour code, MCC/MNC and the carrier frequency of the
  network, which also gives the receiver error (see 6.5). Needs the channel decoding of
  EN 300 392-2 clause 8 (descrambling, deinterleaving, depuncturing, Viterbi, CRC).
- Calibrate the receiver (ppm) against a signal of known frequency and rerun 2.11.
- A recording with a better gain: with +-2 ADC steps the shape, the roll-off and the SNR are weak.

## 6. Step 1 of a TETRA decoder: find the synchronization bursts

Source of the facts below: ETSI EN 300 392-2 V3.8.1 (2016-08), clauses 4.5, 5, 9.4.4 and 9.4.5, text
extracted from the PDF of etsi.org (the standard is not copied into this repository). All sequences
below are bits of the modulation B(m); table 5.1 maps a bit pair to the phase step of a symbol.

### 6.1 What the standard says (quoted figures)

- Table 5.1, pi/4-DQPSK: B(2k-1) B(2k) = 11 -> -3pi/4, 01 -> +3pi/4, 00 -> +pi/4, 10 -> -pi/4;
  S(k) = S(k-1) exp(j dphi(k)). 36 kbit/s, 18 kBd.
- Time structure (4.5.2): slot 255 symbols (510 bits, 14.167 ms), frame 4 slots (56.67 ms),
  multiframe 18 frames (1.02 s, frame 18 is the control frame), hyperframe 60 multiframes.
- Frequency correction field f1..f80 (9.4.4.3.1): 8 ones, 64 zeros, 8 ones. It is an unmodulated
  carrier 2.25 kHz **above the nominal carrier** (32 symbols), with 4 symbols at 6.75 kHz below it at
  both ends. Seen as phase steps: -135, +45, -135 degrees (4 + 32 + 4 symbols).
- Synchronization training sequence y1..y38 (9.4.4.3.4):
  `1,1,0,0,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,1,0,0,1,1,1,0,0,0,0,0,1,1,0,0,1,1,1`.
- Normal training sequence 3: q1..q22 = `1,0,1,1,0,1,1,1,0,0,0,0,0,1,1,0,1,0,1,1,0,1`
  (n1..n22 and p1..p22 for normal training sequences 1 and 2 are in 9.4.4.3.2).
- Synchronization continuous downlink burst (table 9.9, bit numbers 1 to 510): 1 to 12 q11..q22;
  13 to 14 phase adjustment; **15 to 94 frequency correction**; 95 to 214 scrambled synchronization
  block 1 (120 bits, the BSCH); **215 to 252 synchronization training sequence y**; 253 to 282
  scrambled broadcast block (AACH); 283 to 498 scrambled block 2 (216 bits); 499 to 500 phase
  adjustment; 501 to 510 q1..q10.
- Idle slots (9.4.5.1): in continuous transmission the BS sends normal downlink or synchronization
  bursts in all unused downlink slots of the main carrier. That is why a synchronization burst shows
  up in most slots of a carrier. The BSCH is mapped by the rule FN = 18 and (MN + TN) mod 4 = 3 (9.17).
- Carrier raster (the SYSINFO offsets): 25 kHz with an offset of 0, +6.25, -6.25 or +12.5 kHz.

### 6.2 Method (the function `tetra_sync` of the prototype)

Input: the symbols after the matched filter and the timing loop (2.8), derotated by the carrier
offset found in 2.10. Per symbol the phase step d = arg(s_k conj s_(k-1)).

1. **Frequency correction field as a template.** For every start j: score = |sum over 40 symbols of
   exp(j (d - t))| / 40 with t = -135 (4x), +45 (32x), -135 (4x) degrees. The angle of the sum is the
   rotation per symbol that is left: the **carrier error** of that burst, to a few Hz. Random data
   scores 0.14 (median); a perfect match 1.0. Peaks at least 40 symbols apart.
2. **Check the training sequences at the places the burst layout predicts.** From the first symbol of
   the field (bit 15 = symbol 8): y starts 100 symbols later, q11..q22 7 symbols earlier, q1..q10
   243 symbols later. Quantize d to the nearest of +-pi/4, +-3pi/4 (after removing the rotation
   found in 1), map to bits with table 5.1 and count the wrong bits. Random: half of them.
3. Try the conjugate stream as well (I/Q swapped); the wrong orientation gives the random level.

### 6.3 Result for the recording

| | channel 0 | channel 1 | channel 2 |
| --- | --- | --- | --- |
| strong field peaks (score >= 0.8, of about 405 slots) | 301 | 301 | 301 |
| spacing of the peaks, symbols | 255 x180, 510 x91 | 255 x186, 510 x90 | 255 x181, 510 x92 |
| wrong bits of y (38; random 19), mean | 5.5 | 4.9 | 4.3 |
| y exact / at most 4 wrong, of 301 | 53 / 164 | 84 / 178 | 109 / 202 |
| wrong bits of q11..q22 (12; random 6), mean | 1.4 | 1.3 | 1.0 |
| wrong bits of q1..q10 (10; random 5), mean | 1.7 | 1.5 | 1.4 |
| nominal carrier from the field (RF, MHz) | 427.86438 | 427.91438 | 428.08936 |
| conjugate stream, wrong bits of y | 17.9 | 18.0 | 18.0 |

- Every peak is a multiple of 255 symbols from the previous one (a few +-1 are timing jitter), so the
  slot grid is exact. An independent check without the symbol timing: the complex autocorrelation of
  each channel peaks at 906.7 samples of 64 kS/s = 14.17 ms (correlation 0.19 to 0.21).
- The three carriers have their bursts at the **same symbol positions** (146, 401, 912, 1166, ...):
  they are time-synchronous, as the carriers of one cell must be (EN 300 392-2 says the slot, frame
  and multiframe numbering are the same for all carriers of a cell).
- Wrong orientation gives the random level (about 18 of 38), so the signal is as recorded, not I/Q
  swapped.
- **This is TETRA**: a 40-symbol pattern with a random score of 0.14 matches at 0.8 to 0.94 in 300
  slots per carrier, and 100 symbols later the 38-bit sequence y matches with 4 to 5 errors where 19
  are expected. Not decoded yet: the content (BSCH and further).

### 6.4 Pitfalls of this step

1. My first spacing histogram merged neighbouring values (254 and 255 in one bin, labelled by the
   smaller) and showed "254 symbols per slot". Counting exact values gave 255. A summary figure that
   groups values must show the range it grouped.
2. Peaks at low scores (0.45 to 0.8) are many and mostly false; the statistics use score >= 0.8 only.
3. The first symbols of a burst depend on the phase of the previous one: bit pairs are phase *steps*,
   so the first step of y is relative to the last symbol of block 1, which is part of the burst. No
   special handling was needed, but a decoder that works burst by burst has to keep one symbol of
   history.
4. The pi/4 ambiguity of 2.10 is not needed any more here: the field gives the carrier directly.

### 6.5 What the carrier gives

The nominal carriers from the field: 427 864 378, 427 914 375 and 428 089 361 Hz (a few Hz
uncertainty from the tone estimate). Spacing 49.997 and 174.986 kHz. All three are 14.36 to
14.38 kHz above a multiple of 25 kHz. With the raster offsets of 6.1 that means a receiver error of
+1.87 kHz (4.4 ppm, if the offset is +12.5 kHz), -4.38 kHz (-10.2 ppm, offset -6.25 kHz), +8.1 kHz
(18.9 ppm, +6.25 kHz) or +14.4 kHz (33.6 ppm, offset 0). SYSINFO (the BNCH) tells which; the first is
the most plausible for a stick with a small error, but it is a guess until the PDU is decoded.

### 6.6 Consequences for the blocks (4)

- `tetra_burst_find` (or a generic `sync_detect` with a template of phase steps): the template
  match of 6.2 is one `correlate` on a differential signal; it gives burst position, type (field and
  training sequence), orientation, carrier error and a quality figure.
- The frequency correction field is a free, per-burst carrier measurement: `carrier_estimate`
  of 4 should accept a known tone template.
