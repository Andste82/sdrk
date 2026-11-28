# TETRA decoder: what SDRk still lacks

Status: draft, 2026-11-28. Source of every parameter below: ETSI EN 300 392-2 V3.8.1 (2016-08), the
clause is named in each entry. The signal analysis that led here is in
[signal-analysis-notes.md](signal-analysis-notes.md); the recording has three TETRA downlink
carriers whose synchronization bursts were found (section 6 there).

Scope: phase modulation (pi/4-DQPSK), downlink, V+D. Not in scope: uplink bursts, pi/8-D8PSK, QAM
(TEDS), direct mode, speech and encryption (section 4).

## 1. The chain, with what exists

```text
IQ -> ddc -> resample -> pulse_match -> symbol_sync -> psk_demod (pi4) -> soft bits
   -> burst sync (find FCF / training sequences, slot clock)            <- missing
   -> burst demux (cut the fields of the 510 bits)                       <- missing
   -> per logical channel: descramble -> deinterleave -> depuncture
        -> Viterbi -> CRC-16 / RM decode -> type-1 bits                  <- missing
   -> MAC PDU parser (SYNC, ACCESS-ASSIGN, SYSINFO, ...)                 <- missing
```

Present and reusable: `ddc`, `resample`, `pulse_match` (RRC 0.35), `symbol_sync`, `psk_demod` with
`variant: pi4` and `differential`, `correlate` (the FFT correlator, M50), the soft-bit convention
(above 0 is a 1), controllers in the style of `LockWatch`, JSON text packets as in `adsb_parse`.
Checked in the code and in M40.1: the mapping of `psk_demod` / `symbol_map` with `variant: pi4` is
`00 -> +pi/4, 01 -> +3pi/4, 11 -> -3pi/4, 10 -> -pi/4` with the first bit first, which is table 5.1
of the standard (B(2k-1) B(2k)). M40.1 tested it against the DQPSK theory (BER within 1.06 times).
What is left is the test on the recording: the synchronization training sequence must come out
with 0 to 4 wrong bits.

## 2. Missing kernels and blocks

Sizes are relative: S = a day of work and a test, M = a few days, L = a milestone of its own.

| # | Name | Kind | Clause | What | Size |
| --- | --- | --- | --- | --- | --- |
| 1 | `sync_detect` | block (backlog item) | 9.4.4.3 | Template match on the **phase steps** of the symbols: the frequency correction field (-135 x4, +45 x32, -135 x4 degrees) and the training sequences y (38 bits), n1/n2 (22), q (22) as bit patterns. Output per match: symbol index, burst type, score, **carrier error in Hz** (the angle of the correlation), orientation (I/Q swapped or not). The prototype is `tetra_sync` of the analysis prototype (a coherent sum over 40 symbols; [signal-analysis-notes.md](signal-analysis-notes.md) §6.2). | M |
| 2 | `tdma_clock` | block | 4.5, 9.3.5 | Slot (255 symbols), frame (4 slots), multiframe (18 frames) and hyperframe (60) counters TN, FN, MN, locked to the bursts of 1 and kept through slots without a field (a flywheel with a miss counter); the numbers come from the SYNC PDU (TN, FN, MN). Predicts the start of the next slot. | M |
| 3 | `burst_demux` | block | 9.4.4.2 tables 9.7, 9.9 | Cuts the 510 bits of a slot into fields by a table per burst type: block 1 (216), block 2 (216), broadcast bits (30, in two parts 14 + 16 around the training sequence of a normal burst), synchronization block 1 (120), training sequences, phase adjustment. Output: the soft bits of each field as a packet or a lane with a tag. Table-driven so that other burst types are data, not code. | S |
| 4 | `lfsr_scramble` (additive) | kernel + block | 8.2.5 | The scrambling sequence p(k): a 32-bit register with the connection polynomial c(x) = 1 + x + x^2 + x^4 + x^5 + x^7 + x^8 + x^10 + x^11 + x^12 + x^16 + x^22 + x^23 + x^26 + x^32, initial state p(k) = e(1-k) for k = -29..0 and p(-31) = p(-30) = 1, restarted for every block of bits; b5(k) = b4(k) + p(k). e(1..30) is the extended colour code (MCC 10 + MNC 14 + colour code 6 bits, figure 23.5); **all zero for the BSCH**. Soft input (sign flip), hard variant for the encoder. `bit_scramble` and `Prbs` do not fit (multiplicative, two taps). Parameters: taps, init, `scrambling_code` as a **runtime** parameter (set after the SYNC PDU is decoded). | S |
| 5 | `block_deinterleave` | kernel + block | 8.2.4.1 | The (K, a) block interleaver: b4(k) = b3(i) with k = 1 + ((a i) mod K). Used with (120, 11) BSCH, (216, 101) SCH/HD, BNCH, STCH, half-slot speech, (432, 103) SCH/F and full-slot speech. No table needed (a multiply and a modulo per bit, or a permutation built at construction). Interleaver and deinterleaver (the encoder is needed for tests). Interleaving over N blocks (8.2.4.2) later. | S |
| 6 | `depuncture` | kernel | 8.2.3.1.2 to .6 | From the K3 received bits to the 4 K2 bits of the mother code: b3(j) = V(k), k = 8 ((i-1) div t) + P(i - t ((i-1) div t)); the missing positions get a soft value of 0 (erasure). Rate 2/3: t = 3, P = (1, 2, 5), i = j; rate 1/3: t = 6, P = (1, 2, 3, 5, 6, 7); 292/432: t = 3, i = j + (j-1) div 65; 148/432: t = 6, i = j + (j-1) div 35. The first milestone needs 2/3 only. | S |
| 7 | `viterbi` | **kernel with SIMD** + block | 8.2.3.1.1 | Soft-decision decoder of the 16-state mother code of rate 1/4 (constraint length 5), generators G1 = 1 + D + D^4, G2 = 1 + D^2 + D^3 + D^4, G3 = 1 + D + D^2 + D^4, G4 = 1 + D + D^3 + D^4, bits V(4(k-1) + i). Termination: the four tail bits are zero, so the decoder ends in state 0. Parameters: constraint length, generators, number of outputs per input, tail-terminated or not. The same kernel serves the DAB/DVB item in the backlog (K = 7), where the **16 states are one vector of `i16` path metrics** at TETRA and the metrics are Q15-friendly. Output: hard bits (and the re-encoded bit error count as a quality figure). The largest single item. | L |
| 8 | `crc16_bits` | function in `sdrk-proto` | 8.2.3.3 | The (K1 + 16, K1) block code on a **bit** string of any length: generator X^16 + X^12 + X^5 + 1, the register starts with ones and the parity is complemented (equations 8.14 to 8.18, "see ITU-T X.25"); K1 = 60 (BSCH), 124 (SCH/HD), 268 (SCH/F). `crc16_x25` takes whole bytes and the bit order of HDLC; the lengths here are not multiples of 8. | S |
| 9 | `rm_decode` | kernel + block | 8.2.3.2 | Maximum-likelihood decoding of the shortened Reed-Muller code (30, 14) of the AACH (generator matrix in the standard): 2^14 codewords against 30 soft bits, or a fast transform. A generic small linear block code decoder with the matrix as a parameter (n up to 32, k up to 16) does it. | S |
| 10 | `bitfield_parse` | block | clause 21 | The MAC PDUs are tables of fields (bit length, name, values). A table-driven parser instead of code per PDU: output JSON text packets, as `adsb_parse`. First tables: **SYNC PDU** (60 bits: system code 4, colour code 6, timeslot 2, frame 5, multiframe 6, sharing mode 2, TS reserved frames 3, U-plane DTX 1, frame 18 extension 1, reserved 1, TM-SDU 29 with MCC 10 and MNC 14 in D-MLE-SYNC, clause 18), **ACCESS-ASSIGN** (14 bits, AACH, clause 21.4.7.2), **SYSINFO** (BNCH, clause 21.4.4.1, table 21.65: main carrier 12 bits, frequency band 4, offset 2 with 0 / +6.25 / -6.25 / +12.5 kHz, duplex spacing 3, reverse operation 1, ...). Later: MAC-RESOURCE, MAC-DATA, MLE and CMCE messages. | M |
| 11 | `tetra_cell` (controller) | composite / state | 23.2.1 | Learns MCC, MNC and colour code from the first good SYNC PDU, sets the extended colour code of 4 for all other channels, owns the slot clock of 2, counts per channel the blocks with CRC OK, and restarts the acquisition when the CRC OK rate drops (compare the lock watch of `qam_demod`). | M |
| 12 | `tetra_frame_decode` | composite | whole chain | The composite of the stages above, like `qam_frame_decode`: IQ at the channel rate in, packets with the type-1 bits per logical channel (and the parsed PDUs) out, with a map of parameters. | M |
| 13 | `tetra_frame_encode` | composite | clauses 8, 9, 5 | The transmit side for tests only: CRC, tail bits, RCPC encoding, puncturing, interleaving, scrambling, burst assembly, `psk_mod` `pi4` with RRC. Round-trip tests without the real signal, bit error curves, golden vectors of our own. | M |

Order, with a first milestone that needs the least:

1. **BSCH only** (1, 3 for the synchronization burst, 4, 5, 6, 7, 8, 10 for the SYNC PDU): the
   120 coded bits at bit 95 to 214 of a synchronization burst give the 60-bit SYNC PDU with a
   valid CRC. The recording holds about 300 synchronization bursts per carrier, so the success
   rate can be measured and the whole chain is verified by one number: CRC OK. Output: colour code,
   MCC, MNC, timeslot, frame, multiframe numbers.
2. **AACH and SYSINFO** (9, 2, 11, the SYSINFO table of 10): the broadcast block of every slot and
   the system information; the latter contains the carrier frequency and offset, which settles the
   receiver error (6.5 of the notes).
3. **SCH/HD, SCH/F, STCH** (the same stages 4, 5, 7, 8 with the parameters of 216 and 432 bits, and
   the MAC PDU tables of 10): the signalling of the cell. Needs the colour code from 1.
4. 12 and 13 once the stages exist; 13 can come earlier as the test source of 7.

## 3. Facts for the design decisions

- **Soft bits everywhere**: an erasure is a soft value of 0. In the convention "above 0 is a 1,
  0 is a 0" that is also a hard 0, so the depuncturer and `viterbi` must treat exactly 0 as "no
  information" (a branch metric contribution of 0), not as a received 0 bit.
- **The rates in use**: all signalling channels of this document are rate 2/3 with 16 states; the
  other puncturing schemes belong to traffic channels.
- **Scrambling restarts per block**: p(k) starts again for every block; the BSCH uses e = 0.
- **Timing of the channel**: a slot is 14.167 ms; per slot the decoder gets 255 symbols and has
  to decode up to two blocks of 216 coded bits and the 30 bits of the AACH. A Viterbi decoder
  with 16 states does 16 x 2 add-compare-select steps per input bit, about 9000 for a full-slot
  block of 288 steps: negligible on the host even without SIMD. The brute-force decoder of the
  AACH (2^14 codewords x 30 soft bits, about 0.5 million multiply-adds per slot) is the larger
  one, which is why 9 may want a fast transform. The question for the targets is the memory of
  the traceback (16 states x 144 steps for SCH/HD, 288 for SCH/F).
- **Algorithm check against the SDRk goals** (the rule of the project): everything above is `f32`
  on the host first; Q15 and the MCU variants of 4, 5, 6, 7 and 9 are straightforward (integer
  metrics, no divisions) and stay on hold with the other embedded targets.
- **Cycles per decoded bit** is the figure for 4, 5, 7 and 9, against a scalar reference; no
  instruction counts.

## 4. Not in this list, and why

- **Speech (TCH/S)**: the speech codec and the speech channel coding are a separate standard,
  ETSI EN 300 395-2 (ACELP at 4.567 kbit/s, three classes of bits, other punctured codes of 8.2.3.1:
  292/432 and 148/432). A project of its own, and an open question whether it belongs in SDRk.
- **Encryption**: most networks encrypt the air interface (TEA1 to TEA4) once a call or the
  signalling needs it. The broadcast channels (BSCH, AACH, SYSINFO) are not encrypted and are what
  this list targets.
- **Content addressed to others**: decoding speech or messages of other parties raises the
  question of telecommunications secrecy (in Germany the secrecy of telecommunications; not legal
  advice). The proposal stays with the information that every receiver is meant to use (the
  synchronization and system broadcast, and the access assignment, for the purpose of network
  analysis and calibration).
- **Uplink bursts, pi/8-D8PSK, TEDS (QAM)**: other burst types and rates; the table-driven
  `burst_demux` and the generic kernels keep the door open.

## 5. Open points for the maintainer

1. Reference implementations of TETRA exist (osmo-tetra, tetra-kit; GPL). Same rule as for
   rtl_433 (read for protocol facts, two-stage fact sheet, no copying), or the standard only?
   The standard is free and complete for everything in section 2 except the contents of the PDU
   tables of clause 21, which are long but not hard.
2. One `viterbi` kernel for TETRA, DAB and DVB with parameters, or one per code. The proposal is
   one, generic in constraint length (4 to 7) and generators.
3. Whether `tetra_*` blocks are public kinds of the pipeline (a registry entry, a description, a
   schema entry, a count entry) or a crate of their own, as protocols that most users never need.
4. The recording: may a 6-second part (or just the symbols) be added as a regression case? It is
   foreign data; only its derived bursts could be kept.
5. Recording with a better gain (the current one uses +-2 steps of the ADC): the first milestone
   works with the existing one, since the CRC is the check, but the margin is thin.

## 6. Where each item stands in the backlog and the milestones (checked 2026-11-28)

The milestone table in `docs/milestones/README.md` ends at M56; there is no milestone for TETRA or
for any item below. The only places are `docs/milestones/BACKLOG.md` (the lines named here), the
"Analysis blocks" line of `docs/PLAN.md` (line 194) and the same line in the living plan document.
The DAB/DAB+/DVB roadmap in the backlog is explicitly "not scheduled, each step a milestone of its
own".

| # | Item | In the backlog? | Remark |
| --- | --- | --- | --- |
| 1 | `sync_detect` | **Yes**, twice (FFT extensions; "Digital modes after M44": "a sample-level preamble correlator") | Built on `correlate` (M50, implemented). The backlog entry is a peak search with a threshold; the TETRA case also needs a template of **phase steps**, the burst type, and the carrier error as an output. |
| 2 | `tdma_clock` | No | "DMR framing" is named under protocol stacks; DMR is TDMA as well, so the slot clock is shared work if both come. |
| 3 | `burst_demux` | No | |
| 4 | `lfsr_scramble` (additive, 32 taps) | **Partly**: "energy dispersal" of DAB and DVB-T (roadmap steps 5 and 6) is an additive LFSR scrambler, named as a stage only | `Prbs` (two-tap PRBS, M38/M49) and the PN9 whitening exist; `bit_scramble` is multiplicative; the M49 `scramble` is a keyed phase scrambler of complex samples, a different thing. |
| 5 | `block_deinterleave` | **Partly**: other interleavers (DAB time and frequency, DVB-T bit, symbol and convolutional; steps 5 and 6) | The (K, a) block interleaver is not named. |
| 6 | `depuncture` | **Inside** the Viterbi item ("punctured codes of DAB and DVB-T/S") | The TETRA rates (2/3, 1/3, 292/432, 148/432) are not named. |
| 7 | `viterbi` | **Yes**: roadmap step 1, "FEC with SIMD: a soft-decision Viterbi decoder (K = 7, punctured codes of DAB and DVB-T/S) ... SIMD comes before threads"; also "forward error correction (convolutional codes with Viterbi, Reed-Solomon, LDPC)" under digital modes. Not scheduled. | TETRA is K = 5, four outputs per input. One kernel generic in K, generators and outputs per input would serve all. Using TETRA as the first user changes the order of the roadmap: a decision for the maintainer. |
| 8 | `crc16_bits` | No | M37 has the CRCs of whole bytes (`crc16_x25`, `crc16_ccitt_false`, `crc32`, `crc24_adsb`). |
| 9 | `rm_decode` | No | The backlog had Reed-Solomon (DVB, DAB+), not Reed-Muller. `fwht` (M56, **planned**, not implemented) would speed a decoder up. |
| 10 | `bitfield_parse` | No | "Protocol stacks" and parsers are listed one by one (APRS payload parser, ADS-B CPR, device decoders), not a table-driven one. |
| 11 to 13 | `tetra_cell`, `tetra_frame_decode`, `tetra_frame_encode` | No | TETRA appears only as the use of pi/4-DQPSK in M40.1 and in three parameter descriptions. |
| | **Analysis blocks** (`signal_find`, `occupied_bandwidth`, `symbol_rate_estimate`, `power_stats`) | **Yes**: "Analysis blocks with structured results (frames/events): signal detection (frequency, bandwidth, level, SNR), occupied bandwidth, noise floor, burst detection, instantaneous values and histograms, symbol-rate estimator" (BACKLOG, PLAN.md, the living plan) | Not named: `rolloff_estimate`, `modulation_estimate`, `carrier_estimate` with its ambiguity rule, and the template match of 1. |

Done already and used: `psk_mod` / `psk_demod` with `variant: pi4` (M40.1), `pulse_match` and RRC
design (M40, M44.1.1), `symbol_sync` (M38), `correlate` (M50), `channelize` (M45). Planned but not
implemented: the DSSS/OFDM kernels M51 to M56, among them `fwht` (M56).

Suggested way to put it on the list (the maintainer decides): a milestone for a generic `viterbi`
kernel (the shared item of TETRA, DAB and DVB), one for `sync_detect` with the template mode, one
for the channel-coding primitives (additive LFSR scrambler, (K, a) block interleaver, depuncturer,
bit-level CRC-16, Reed-Muller decoder), one for the analysis blocks that the notes prove useful, and
only then the TETRA blocks and composites that use them.

## 7. Generalized in the backlog (2026-11-28)

The items of section 2 that are not specific to TETRA are now one entry of `BACKLOG.md`:
*Generic building blocks for framed digital modes* (`viterbi` with `conv_encode`, `puncture`,
`rs_decode`, `block_code`, `crc`, `interleave`, `lfsr` with `additive_scramble`, `sync_detect`,
`frame_clock`, `field_demux` and `bitfield_parse`), each defined by algorithm and parameters with
TETRA, DAB, DVB and others as presets. The DAB/DVB roadmap refers to them; the analysis blocks entry
lists the steps of the notes; M56 `fwht` names its uses beyond MC-CDMA. What stays TETRA-only is
`tetra_cell`, `tetra_frame_decode` and `tetra_frame_encode` (items 11 to 13), the presets and the PDU
tables.
