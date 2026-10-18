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
| `QuadDemodState` | `1/1` | `1/2`, `Delay::new(1, 2)` (the discriminator looks at two samples) |
| `DeemphasisState` | `1/1` | 0 (IIR; the real delay depends on the frequency and is not part of exact event timing) |
| `DcBlockState` | `1/1` | 0 (one-pole high-pass; the delay depends on the frequency) |

`window`, `mag_sq` and `power_db` are stateless per-sample kernels with ratio `1/1` and no delay. The window generators and the FFT-related timing follow with M07.

The FIR group delay is exact for symmetric (linear-phase) taps. A filter with asymmetric taps still reports `(N - 1) / 2`; it is the declaration of the structure, not a measurement.

## Block-size invariance

Output samples are a function of the absolute input position only, never of how the input was cut into calls: the same input produces bit-identical output whether it arrives in one block or in chunks of any size. `rotate` and `fir` keep their state (`Nco`, `FirState`) between calls and are tested with `check_block_invariance` over random chunk plans; the same holds for `fir_decimate`, `resample`, `quad_demod`, `deemphasis` and `dc_block` (20 plans each).

## Rate-changing kernels

`fir_decimate`, `resample` and `resample_real` return `Progress { consumed, produced }`. They consume input only as far as the output space allows and the caller resubmits `input[consumed..]`; the output depends only on the absolute input position, so the sequence of outputs is the same for any block size and any output buffer size. After `n` input samples since a reset the decimator has produced `ceil(n / M)` outputs and the resampler `floor((n * L - 1) / M) + 1` (see `docs/kernels.md`). A resampler whose output buffer ran full keeps the remaining outputs pending; they are produced by the next call (also with an empty input).

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

Rotating by the cosine/sine of the phase uses a 4096-entry table with linear interpolation; the table is generated at build time (see `docs/kernels.md`).

## Further parts

Block delay accounting in the pipeline, burst timing and timestamps follow with the blocks and pipeline milestones.
