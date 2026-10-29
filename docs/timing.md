# Timing

How SDRk describes the time behaviour of kernels and blocks. Everything here is about *sample counts*; the sample index is the only time base and no kernel reads a clock.

## The `Timing` trait

Every kernel state and every block implements `sdrk_core::Timing`:

| Method | Meaning |
| --- | --- |
| `ratio() -> Ratio` | output rate divided by input rate as an exact reduced fraction (`1/1` for same-rate kernels, `1/10` for decimation by 10) |
| `ratio_kind() -> RatioKind` | `Constant` (default) or `Variable` if the ratio can change at run time |
| `group_delay() -> Delay` | delay between an input sample and its effect on the output, in *input* samples, as an exact fraction |

`Ratio` and `Rate` come from `sdrk-device` and use `u128` integer arithmetic; `Delay` (`num/den`, `Delay::reduced`, `Delay::add`) is exact as well. Floats are never used for time.

Declared values:

| State | Ratio | Group delay |
| --- | --- | --- |
| `Nco` | `1/1` | 0 |
| `FirState` with `N` taps | `1/1` | `(N - 1) / 2`, built as `Delay::new(N - 1, 2)` (not reduced; call `reduced()` before comparing) |
| `FirDecimState` with `N` taps, factor `M` | `1/M` | `(N - 1) / 2`, built as `Delay::new(N - 1, 2)` |
| `ResampleState`, `ResampleRealState` with `N` taps, `L / M` | `L/M` | `(N - 1) / (2 L)`, built as `Delay::new(N - 1, 2 * L)` |
| `FirInterpState`, `FirInterpRealState` with `N` taps, factor `L` | `L/1` | `(N - 1) / (2 L)`, built as `Delay::new(N - 1, 2 * L)` |
| `FmModState` | `1/1` | 0 |
| `QuadDemodState` | `1/1` | `1/2`, `Delay::new(1, 2)` (the discriminator looks at two samples) |
| `DeemphasisState` | `1/1` | 0 (IIR; the real delay depends on the frequency and is not part of exact event timing) |
| `DcBlockState` | `1/1` | 0 (one-pole high-pass; the delay depends on the frequency) |
| `DcBlockRealState` | `1/1` | 0 (one-pole high-pass; the delay depends on the frequency) |
| `AgcState` | `1/1` | 0 (the gain acts on the sample it is applied to; the loop only decides the next gain) |
| `SquelchState` | `1/1` | 0 (the gate acts on the sample it looks at; the average power only decides, it does not delay the signal) |

`window`, `mag_sq`, `mag` and `power_db` are stateless per-sample kernels with ratio `1/1` and no delay.

The FIR group delay is exact for symmetric (linear-phase) taps. A filter with asymmetric taps still reports `(N - 1) / 2`; it is the declaration of the structure, not a measurement.

## Block-size invariance

Output samples are a function of the absolute input position only, never of how the input was cut into calls: the same input produces bit-identical output whether it arrives in one block or in chunks of any size. The one exception are the `f32`/`C32` recursions `deemphasis`, `dc_block` and `dc_block_real` of the SIMD backend, whose state is rounded to `f32` at every call boundary: their bits depend on the split, within the regression guards of their golden cases (maintainer decision of 2026-10-29, [kernels.md](kernels.md#backends-and-the-numeric-contract)); their timing is unaffected. `rotate` and `fir` keep their state (`Nco`, `FirState`) between calls and are tested with `check_block_invariance` over random chunk plans; the same holds for `fir_decimate`, `resample`, `quad_demod`, `deemphasis`, `dc_block`, `dc_block_real`, `agc`, `squelch`, `biquad` and `biquad_real` (20 plans each; the average power, the gate and the section histories are part of the state).

The SIMD `rotate` for `C32` ([backends.md](backends.md)) cuts the stream into blocks of `B` samples (a power of two: 512 at SSE, NEON and SIMD128, 1024 at AVX2, 2048 at AVX-512, or the period of the phase sequence if that is shorter) at absolute positions. The position of a sample in its block is not stored: with `inc = 2^t o` (`o` odd) it is `((phase >> t) * o^-1) mod B`, a function of the phase and the increment alone. Every quantity a call computes once (lane offsets, table entries, step) is a function of the increment alone, and table entries are computed only for the positions a call covers but always by the same formula. So every output sample is a function of `(phase_n, inc)` and the input sample, and the output is bit-identical for any split into calls. Its bits can depend on the increment as well as on the phase: after a retune (`set_frequency`) the sample whose phase was still advanced with the old increment can differ from an unchanged run by rounding (about 1e-7). It is numerically still the old-frequency rotation, and the new frequency acts from the next sample on. The pipeline test `time_events.rs` (`a_timed_change_is_exact_at_decimation_1`) therefore checks the first output that changes by more than 1e-6 of the peak (the real change is 8.7e-6), not the first bit difference.

## Rate-changing kernels

`fir_decimate`, `resample` and `resample_real` return `Progress { consumed, produced }`. They consume input only as far as the output space allows and the caller resubmits `input[consumed..]`; the output depends only on the absolute input position, so the sequence of outputs is the same for any block size and any output buffer size. After `n` input samples since a reset the decimator has produced `ceil(n / M)` outputs and the resampler `floor((n * L - 1) / M) + 1` (see `docs/kernels.md`). A resampler whose output buffer ran full keeps the remaining outputs pending; they are produced by the next call (also with an empty input).

## Transmit timing and lead time

The transmit chain is the receive chain reversed: the interpolating filters delay the signal, so the first useful output sample appears later than the first input sample, and a burst that has to start at an exact sample index needs the *lead time* of the whole chain. The lead time (in output samples of the last stage) is the sum of the group delays of the interpolation filters, each converted to the output rate: a stage with group delay `d` (in its own input samples) contributes `d * R` output samples, where `R` is the total interpolation from the input rate of that stage to the output rate. For the FM transmitter of the end-to-end test (`fir_interpolate` by 5 with 241 taps, `fm_mod`, `duc` by 10 with 127 taps) this is `(240 / 10) * 50 + (126 / 20) * 10 = 1200 + 63 = 1263` output samples at 2.4 MHz (the audio filter dominates: 24 audio samples times 50). `Delay::add` builds such sums exactly; the pipeline does the accounting automatically (`Pipeline::lead_time`, see [Transmit: lead time, bursts, index](#transmit-lead-time-bursts-index)).

`fm_mod` has no delay (`Delay::ZERO`); its phase accumulator and the oscillator of a `duc` are part of the state and advance by exactly the number of samples processed, never by the number of samples offered. The fused `ddc` and `duc` kernels keep that property when the output buffer is too small (see `docs/kernels.md`).

## The NCO phase model

`Nco` holds a 32-bit phase accumulator `phase` in units of 2π/2^32 and an increment `inc`:

```text
phase_n     = phase_0 + n * inc   (mod 2^32)
inc         = round(frac(f / fs) * 2^32)   (mod 2^32, round half to even)
frac(x)     = x - floor(x)
output[n]   = input[n] * exp(j * 2 * pi * phase_n / 2^32)
```

- Positive frequencies move a spectrum up. To bring a signal at +f to 0 Hz, rotate by −f.
- Negative frequencies map to `inc > 2^31`; `f = 0.25 fs` gives `inc = 2^30`, `f = -0.25 fs` gives `3 * 2^30`. Frequencies of one sample rate or more alias (`frac` drops whole turns).
- The phase wraps naturally with `u32` arithmetic, so there is no drift and no accumulated floating-point error however long the stream runs. The frequency resolution is `fs / 2^32` (about 0.23 mHz at 1 MS/s).
- `Nco::set_frequency` changes only `inc` and keeps the phase: the signal stays phase-continuous across a retune. `set_phase` and `reset` are the only ways to jump.
- A call to `rotate` advances the phase by exactly `input.len()` samples (also when called with an empty slice, which changes nothing).
- Frequencies are given in hertz together with the sample `Rate`; NaN and infinity are rejected with `InvalidParameter`.

In the fallback, rotating by the cosine/sine of the phase uses a 4096-entry table with linear interpolation; the table is generated at build time (see `docs/kernels.md`). The reference evaluates `cos` and `sin` of the phase in f64. The SIMD backend evaluates the `C32` oscillator per block instead (the block oscillator of [backends.md](backends.md), see [Block-size invariance](#block-size-invariance)); its `CQ15` `rotate` does the table lookup of the fallback, bit-exact.

## Blocks

Every block of `sdrk-blocks` reports its timing as `BlockTiming { ratio, kind, delay }` (see [blocks.md](blocks.md)); the values are exact fractions, derived from the kernel states, and tested against an impulse (peak position within 0.5 input samples) and against measured output counts.

| Block | Ratio (out/in) | Kind | Group delay in input samples |
| --- | --- | --- | --- |
| `convert` | samples per byte of a packing group (for example `1/2` for `cu8`, `2/5` for `packed10`) | `Constant` | 0 |
| `convert_out` | bytes per sample (the inverse) | `Constant` | 0 |
| `scale`, `dc_block`, `fm_mod`, `squelch`, `agc`, `biquad` | `1/1` | `Constant` | 0 (`dc_block`, `biquad`: the delay of the IIR filter depends on the frequency and is not declared) |
| `fir` with `N` taps | `1/1` | `Constant` | `(N - 1) / 2` |
| `ddc` with `N` taps, decimation `M` | `1/M` | `Constant` | `(N - 1) / 2` |
| `duc` with `N` taps, interpolation `L` | `L/1` | `Constant` | `(N - 1) / (2 L)` |
| `fm_demod`, audio decimation `M`, audio filter of `N` taps | `1/M` | `Constant` | `1/2 + (N - 1) / 2` (`1/2` for `M = 1`, which has no audio filter) |
| `am_demod`, audio decimation `M`, audio filter of `N` taps | `1/M` | `Constant` | `(N - 1) / 2` (0 for `M = 1`, which has no audio filter; the delay of the DC blocker depends on the frequency and is not declared) |
| `ssb_demod`, audio decimation `M`, filter of `N` taps | `1/M` | `Constant` | `(N - 1) / 2` (the delay of `fir_decimate`, as `Delay::new(N - 1, 2)`; the two oscillators add none) |
| `resampler`, rational `L/M`, `N` taps at `L` times the input rate | `L/M` | `Constant` | `(N - 1) / (2 L)` |
| `resampler`, tunable, `K` taps per phase | nominal `rate_out / rate_in` | `Variable` | `(128 K - 1) / 256` (the delay of the prototype filter at 128 times the input rate; a `ppm` correction does not change the declaration) |

The `delay` of a decimating block is measured in *input* samples, so an impulse that enters at input index `i` leaves the output grid at the output sample nearest to `i + delay` (the tests align the impulse with the output grid before they compare). `fm_demod` adds the half sample of the discriminator (it looks at two samples) to the delay of the audio filter. The spectrum block has no timing: it emits frames, and `FrameMeta.first_sample` names the input index of the first sample of the frame's first window.

The rational resampler outputs `floor((n L - 1) / M) + 1` samples for `n` input samples after a reset. The tunable resampler puts output `j` at the 32.32 position `j * step` (input samples, the first output at input sample 0, history of zeros in front) and emits it as soon as input sample `floor(pos)` has arrived (sample `floor(pos) + 1` when the phase is the last one of the bank). A million samples at 48 kHz with `ppm = +100` give `1 000 100` outputs (plus or minus 1).

## Index domains in the pipeline

The pipeline never reads a clock; the sample index is the time base. Every stream (pipeline input, output of a node, pipeline output) has its own `u64` index at its own rate. The index of an input starts at the `start` of the first `BlockMeta` that is pushed (joining a stream in the middle), or at 0 with plain `push`.

**Mapping rule.** A node with the exact rate ratio `r = L/M` (from `timing()`) maps the input index `n` to the first output sample whose computation includes it:

```text
m = n0_out + ceil((n - n0_in) * L / M)
```

`(n0_in, n0_out)` is the origin pair of the node: `n0_in = start`, `n0_out = ceil(start * L / M)` initially, and new values after a reset or a gap. Mapping through a chain applies the rule node by node. A block whose ratio is not exactly constant overrides `Block::map_index(rel_in)` (input samples since the last origin to output samples since the origin); the default is `ceil(rel_in * L / M)`. The tuned resampler counts the outputs `j` whose 32.32 position `j * step` lies before `rel_in * 2^32`, in `u128`, always relative to the origin so that nothing overflows at large indices. The rule is what the blocks do: after `N` input samples every node has produced `ceil(N * L / M)` samples (tested for `convert`, `ddc` and `fm_demod`; `crates/blocks/sdrk-blocks/tests/map_index.rs` for the blocks). Blocks count in elements (a raw stream has one element per byte), the pipeline converts the counts at the edges of raw streams so that indices are always in samples.

## Gaps

`push_with_meta` with a start above the expected index inserts a gap of `missing = start - expected` samples. Everything that depends on the input is reset. The origins move: the new input origin of the first node is `start`, and its new output origin is the end of what it had produced plus `ceil(missing * L / M)`; the output gap is the input gap of the next node, and so on. The output indices therefore jump by the mapped gap, and the data that follows equals the output of a pipeline that was reset (`reset()`) at that point. A `Gap { missing }` event, with `missing` counted at the input, is reported at the new output index of every output that depends on the input. The same arithmetic places the first block of an input (joining), without the event.

## Anchors

A `TimeAnchor { index, unix_ns }` on an input says that sample `index` was taken at that absolute time. On a downstream stream the anchor must name the time of the *signal* in the sample, not of the arrival of the sample; filters delay the signal. The anchor `(n, t)` on the input of a node becomes `(m, t')` on its output with `m` from the mapping rule and

```text
t' = t + ((m - n0_out) * M / L - (n - n0_in) - D) / rate_in * 1e9 ns
```

where `D` is the group delay of the node in its input samples (an exact `Delay`) and `rate_in` the rate at its input. The arithmetic is exact in `i128` with round half up. Applying this node by node gives the total delay compensation, including the rounding of the index. `Pipeline::time_of(output, index)` uses the latest anchor of the output (`TimeAnchor::time_of` with the rate of the output): for the FM receiver the time of an audio sample differs from the analytic time of the tone by less than half an audio sample (tested with the analytic FM generator of the testkit, de-emphasis off; the de-emphasis filter is a recursive filter whose phase delay is not part of a group delay).

## Timed parameters

`set_param_at` maps the target to the input index of the node, ends the next `process` call there, sets the parameter and continues. The change is visible at the first output sample computed after the set, and the `ParamApplied` event is at the mapped index of that sample. For a decimating node the first changed output sample is within one sample of the mapped index; for decimation 1 the oscillator of the `ddc` advances sample `n` with the old frequency, so the first changed output is `n + 1` (the output is bit-identical to a block that was split by hand at `n`; tested).

## Latency

`latency(output)` returns `Latency { group_delay, max_buffered }`; `Report.outputs` has the same figures per output.

- `group_delay` is the sum of the group delays of the nodes on the path from the input to the output, each converted to samples of the output: a node with delay `d` in its own input samples and the rates `rate_in(node)` and `rate_out(output)` contributes `d * rate_out / rate_in(node)`. It is an exact fraction (`Delay`). For a frames output the unit is the sample at the input of the frame node, and the frame node adds nothing.
- `max_buffered` is the number of samples that the nodes may hold back because they cannot start with fewer than `min_input` samples: `min_input - 1` per node, converted to samples of the output like the delay and rounded up, summed. The buffers of the outputs themselves are not counted: they hold results that were already delivered.

Together they bound how late a sample can appear: the signal that entered at input index `n` appears in the output sample `m = map(n)` and is complete `group_delay` output samples later; up to `max_buffered` samples of the newest input may not have produced output yet.

## Transmit: lead time, bursts, index

The lead time of a transmit path is `latency(output).group_delay` (`Pipeline::lead_time`). A burst of `N` audio samples gives `N * R` output samples (`R` the total interpolation) plus the tail `ceil(lead_time)`: `end_burst` feeds `ceil(D_in)` zero samples (the delay of the path in input samples) to flush the filters and marks the burst end `ceil(lead_time)` samples after the sample that the last real input leads to (what the zero samples produce beyond is discarded). After the burst the nodes of the path are reset, so the next burst starts from silence with the same output as the first one. The output index continues across bursts and never goes back. `begin_burst(output, tx_time_ns)` marks the first sample of the burst, and a transmit driver that needs the burst to start at `tx_time_ns` is handed the block with `BURST_START` and that time.
