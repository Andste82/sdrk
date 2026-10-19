# Blocks

`sdrk-blocks` (layer `crates/blocks`, `no_std`, no heap) turns kernels into blocks that think in parameters with units ("DDC to 600 kHz, decimate by 10, cutoff 100 kHz") and describe themselves. A block translates its parameters into kernel states, owns no memory of its own and calls the facade `sdrk::Sdrk` for every kernel, so backends speed up the blocks automatically. The pipeline ([pipeline.md](pipeline.md)) builds blocks from JSON with the same names and ranges that are written down here.

## The block interface

```rust,ignore
pub trait Block: Send {
    type In: Copy + Send;
    type Out: Copy + Send;
    fn description(&self) -> &'static BlockDescription;
    fn process(&mut self, input: &[Self::In], output: &mut [Self::Out]) -> Result<Progress>;
    fn timing(&self) -> BlockTiming;               // { ratio, kind, delay }
    fn reset(&mut self);
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
    fn param(&self, name: &str) -> Result<ParamValue>;
    fn min_input(&self) -> usize;
    fn map_index(&self, rel_in: u64) -> u64;       // default: ceil(rel_in * L / M)
}
```

- `process` follows the contract of the rate-changing kernels: it consumes as much input as it can and produces at most `output.len()` samples; the caller resubmits `input[consumed..]`. A call that can neither consume nor produce returns `Progress { 0, 0 }` (no input, no output room, or less than one packing group). A resampler that ran out of output room keeps the remaining outputs pending and delivers them in the next call, also with an empty input.
- The output depends only on the absolute input position, never on how the input was cut into calls or how big the output buffer was (tested for every block with random chunk plans, including 1-sample chunks).
- `map_index(rel_in)` maps a count of input samples since the last origin (construction or `reset`) to the number of output samples that include them: the first output sample that includes input sample `n` is `map_index(n)`. The default is `ceil(rel_in * L / M)` with the exact ratio of `timing()`; the tunable resampler overrides it with its position step (`ceil(rel_in * 2^32 / step)` in `u128`). The pipeline uses it for the index domains of [timing.md](timing.md#index-domains-in-the-pipeline).
- `timing()` is exact: `ratio` is output rate divided by input rate (`Ratio`), `kind` is `Constant` or `Variable` (the tunable resampler), `delay` the group delay in input samples as an exact fraction (see [timing.md](timing.md)).
- `reset()` clears all history (the pipeline calls it on gaps) and keeps the parameters; the output after a reset equals that of a fresh block.
- `set_param` changes a *runtime* parameter from the first sample of the next `process` call. Errors: `UnknownParameter` (name not in the description), `ReadOnlyParameter` (a build parameter), `WrongParamType`, `OutOfRange { name, min, max, got }` (also for NaN). A rejected value changes nothing. `param` returns the value in use, also for build parameters and for defaults that depend on the sample rate (`cutoff_hz` of a `ddc` reads `0.4 * fs / decimation`).
- Blocks that output frames instead of a stream (the spectrum) implement `FrameBlock`: `push(input, frame) -> (consumed, Option<FrameMeta>)` completes at most one frame per call and stops consuming right after it. `FrameMeta { first_sample, averaged, exponent }` (`exponent` is always 0: the block applies the exponent of the block-floating-point FFT itself).

`ParamValue` is `F64`, `I64`, `Bool`, `Choice(index)` or `Format(SampleFormat)`.

## Construction

Every block `Foo` follows the same pattern:

```rust,ignore
pub struct FooConfig { /* pub fields, units in the names: frequency_hz, cutoff_hz, decimation */ }
impl Default for FooConfig { /* the typical values */ }
impl FooConfig {
    pub fn validate(&self, rate_in: Rate) -> Result<()>;
    pub fn set(&mut self, name: &str, value: ParamValue) -> Result<()>;   // by description name
    pub fn get(&self, name: &str) -> Result<ParamValue>;
}
impl<T, S: Storage> Foo<T, S> {
    pub fn new(config: &FooConfig, rate_in: Rate, storage: &mut S) -> Result<Self>;
    pub fn storage_bytes(config: &FooConfig, rate_in: Rate) -> Result<usize>;
    pub fn rate_out(config: &FooConfig, rate_in: Rate) -> Result<Rate>;
}
```

`validate` applies the limits that depend on the input rate (`frequency_hz` within plus or minus half the rate, a stop band below half the rate). Parameters with a rate-dependent default (`cutoff_hz` and `transition_hz` of `ddc`, `duc` and `resampler`) are `Option<f64>` in the config: `None` selects the default. `FooConfig::get` of an unset one is an error; the block reports the resolved value. `set` and `get` work on build and runtime parameters and check kind and range with the same table (`ParamDesc::check`) that the description publishes.

Blocks exist for the complex sample types `C32` and `CQ15` (`ConvertBlock<T, S>`, `ConvertOutBlock`, `ScaleBlock`, `DcBlockBlock`, `FirBlock`, `DdcBlock`, `DucBlock`, `FmDemodBlock` (`C32`/`CQ15` in, `f32`/`Q15` out), `FmModBlock` (`f32`/`Q15` in, `C32`/`CQ15` out), `ResamplerBlock`, `SpectrumBlock`) and for the real types in `ResamplerRealBlock<R, S>` (`f32`, `Q15`), which is the resampler for audio.

## Storage

Blocks never allocate. They ask a `Storage` for zero-initialised buffers while they are built:

```rust,ignore
pub trait Storage {
    type Buf<T: bytemuck::Pod + Default + Send>: AsRef<[T]> + AsMut<[T]> + Send;
    fn alloc<T: bytemuck::Pod + Default + Send>(&mut self, len: usize) -> Result<Self::Buf<T>>;
}
```

- **`ArenaStorage`** carves the buffers out of one caller-provided byte slice by bump allocation (`Buf<T> = &'a mut [T]`, no `unsafe`: the padding comes from `align_offset`, the cast from `bytemuck::try_cast_slice_mut`). It is the storage of microcontrollers, the suite and the examples. A request that does not fit is `Error::Storage { needed, available }` (kind `CapacityExceeded`) and changes nothing.
- **`Vec` storage** is what the upper layers use; `sdrk-pipeline` implements `Storage` with `Buf<T> = Vec<T>` and the testkit has `VecStorage` (feature `std`) for tests.
- **`storage_bytes(config, rate_in)`** is exact for an `ArenaStorage` whose slice starts at a 16-byte aligned address: the buffers are allocated in a fixed order and each starts at the next multiple of its element alignment. `ARENA_ALIGN` is 16 and `AlignedBytes<N>` is a `#[repr(C, align(16))]` byte array for stack arenas and statics. A slice that is not aligned may need up to 15 bytes more. The tests build every block in an arena of exactly `storage_bytes` bytes and also check that one byte less fails with a `Storage` error.
- All big data lives in the storage; the block structs themselves hold a few slices and counters and are at most 512 bytes (tested).

```rust
use sdrk::{Rate, C32};
use sdrk_blocks::{AlignedBytes, ArenaStorage, Block, FirBlock, FirConfig};

let cfg = FirConfig { cutoff_hz: 4_000.0, transition_hz: 2_000.0, ..FirConfig::default() };
let rate = Rate::hz(48_000)?;
let bytes = FirBlock::<C32, ArenaStorage<'_>>::storage_bytes(&cfg, rate)?;
let mut mem = AlignedBytes([0u8; 4096]);
let mut arena = ArenaStorage::new(&mut mem.0[..bytes]);
let mut fir = FirBlock::<C32, _>::new(&cfg, rate, &mut arena)?;
let mut out = [C32::default(); 1];
fir.process(&[C32::new(1.0, 0.0)], &mut out)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Filter design

`sdrk_blocks::design` computes in `f64` and writes into caller slices; the functions mirror SciPy so that golden vectors can check them (`tests/golden/sdrk_golden/design.py` calls SciPy and is the reference):

- `firwin_lowpass(cutoff_hz, fs_hz, window, taps)` is `scipy.signal.firwin(N, cutoff, window=…, pass_zero=True, scale=True, fs=fs)` with a symmetric Hann, Hamming, Blackman or Kaiser window and a DC gain of 1 (`0 < cutoff < fs / 2`, else `OutOfRange`; a window that leaves no DC gain, such as a two-point Hann window, is `InvalidParameter`).
- `kaiserord(ripple_db, transition_hz, fs_hz)` is `scipy.signal.kaiserord(ripple, 2 * transition / fs)` with the tap count forced odd; `kaiser_beta(a)` is `scipy.signal.kaiser_beta`.
- `to_f32` and `to_q15` convert the taps; `to_q15` refuses a tap above `1 - 2^-15` by more than 1 % (it would saturate badly).
- `DesignSpec { cutoff_hz, transition_hz, attenuation_db }` is the parameter set of the filtering blocks. A block designs `kaiserord` + `firwin_lowpass(Kaiser)` one tap at a time (no temporary `f64` buffer), at most `max_taps` taps (default 1023, `OutOfRange` beyond).

The golden tests compare the taps with SciPy to 1e-12 and the tap counts exactly.

## Descriptions

Every block has a `BlockDescription` (kind, summary, details, ports, parameters, required parameters) as `static` data; `all_descriptions()` lists all eleven. A `ParamDesc` has a name that ends in its unit (`_hz`, `_db`, `_us`), a `Unit`, a `ParamKind` with absolute limits and a default, a `typical` text, `Mutability::Build` or `Runtime`, and a description. Rules, all enforced by `crates/blocks/sdrk-blocks/tests/describe.rs`:

- every text is non-empty and has no leading or trailing whitespace; a summary is one sentence of at most 120 characters that ends with a period;
- block kinds and parameter names are `snake_case` and unique; a unit suffix in the name matches the `Unit`;
- defaults lie inside the range; a parameter without a default is either listed in `required` or documents a rate-dependent default (its `typical` or `description` text says "default");
- the config defaults equal the description defaults, `set` and `get` accept and return every described parameter, and every runtime parameter is accepted by `set_param` and readable by `param`;
- `min`/`max` are the absolute limits a JSON schema can check; limits that depend on the sample rate are applied by `validate` and `new`.

The tables below are generated: `cargo test -p sdrk-blocks --test describe -- --ignored print --nocapture` prints them and the test `docs_blocks_md_contains_the_generated_table` fails when this file differs.

<!-- blocks-table:begin -->

| Kind | Summary | Inputs | Outputs | Required |
| --- | --- | --- | --- | --- |
| `convert` | Converts raw device bytes in a sample format to complex samples. | `in`: raw | `out`: c32, cq15 | `format` |
| `convert_out` | Converts complex samples to raw device bytes in a sample format. | `in`: c32, cq15 | `out`: raw | `format` |
| `scale` | Multiplies a complex signal by a gain given in decibels. | `in`: c32, cq15 | `out`: c32, cq15 | - |
| `dc_block` | Removes the DC offset of a complex signal with a one-pole high-pass. | `in`: c32, cq15 | `out`: c32, cq15 | - |
| `fir` | Low-pass FIR filter designed from cutoff, transition width and attenuation. | `in`: c32, cq15 | `out`: c32, cq15 | `cutoff_hz`, `transition_hz` |
| `ddc` | Shifts a signal to 0 Hz, low-pass filters it and decimates it. | `in`: c32, cq15 | `out`: c32, cq15 | `decimation` |
| `duc` | Interpolates a signal, removes the images and shifts it up to a carrier frequency. | `in`: c32, cq15 | `out`: c32, cq15 | `interpolation` |
| `fm_demod` | Demodulates an FM signal to audio with de-emphasis and decimation. | `in`: c32, cq15 | `audio`: f32, q15 | - |
| `fm_mod` | Frequency-modulates a real signal onto a complex baseband carrier. | `in`: f32, q15 | `out`: c32, cq15 | - |
| `resampler` | Changes the sample rate by a rational factor, optionally tunable in ppm. | `in`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | `rate_out_hz` |
| `spectrum` | Computes averaged power spectra in dB relative to full scale as frames. | `in`: c32, cq15 | `frame`: frame(fft_size) | - |

| Block | Parameter | Unit | When | Type, range, default | Typical |
| --- | --- | --- | --- | --- | --- |
| `convert` | `format` | - | build | sample format name, no default | cu8 for RTL-SDR, cs16 for HackRF-class devices |
| `convert_out` | `format` | - | build | sample format name, no default | cs8 for HackRF-class transmitters, cs16 for others |
| `scale` | `gain_db` | dB | runtime | f64, -120 to 40, default 0 | 0 for unity gain, -6 to halve the amplitude |
| `dc_block` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, default 10 | 10 for a 2.4 MHz receiver |
| `fir` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | 15e3 for an audio-bandwidth filter at 48 kHz |
| `fir` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | 0.1 times the cutoff |
| `fir` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `fir` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023 |
| `ddc` | `frequency_hz` | Hz | runtime | f64, -1000000000000 to 1000000000000, default 0 | 600e3 for a carrier 600 kHz above the tuner centre |
| `ddc` | `decimation` | - | build | integer, 1 to 1024, no default | 10 to go from 2.4 MHz to 240 kHz |
| `ddc` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | 100e3 for FM broadcast; default 0.4 times the output rate |
| `ddc` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the output rate |
| `ddc` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `ddc` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023 |
| `ddc` | `max_chunk` | samples | build | integer, 1 to 1048576, default 4096 | 4096 |
| `duc` | `frequency_hz` | Hz | runtime | f64, -1000000000000 to 1000000000000, default 0 | 600e3 to transmit 600 kHz above the centre |
| `duc` | `interpolation` | - | build | integer, 1 to 1024, no default | 10 to go from 240 kHz to 2.4 MHz |
| `duc` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | 100e3 for FM broadcast; default 0.4 times the input rate |
| `duc` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the input rate |
| `duc` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `duc` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023 |
| `duc` | `max_chunk` | samples | build | integer, 1 to 1048576, default 4096 | 4096 |
| `fm_demod` | `deviation_hz` | Hz | build | f64, 1 to 1000000000000, default 75000 | 75e3 for broadcast FM, 5e3 for narrow-band voice |
| `fm_demod` | `deemphasis_us` | us | build | f64, 0 to 1000000, default 50 | 50 in Europe, 75 in North America, 0 to switch off |
| `fm_demod` | `audio_cutoff_hz` | Hz | build | f64, 1 to 1000000000000, default 15000 | 15e3 for broadcast FM |
| `fm_demod` | `audio_decimation` | - | build | integer, 1 to 64, default 1 | 5 to go from 240 kHz to 48 kHz |
| `fm_mod` | `deviation_hz` | Hz | runtime | f64, 0 to 1000000000000, default 75000 | 75e3 for FM broadcast, 5e3 for narrow-band voice |
| `resampler` | `rate_out_hz` | Hz | build | f64, 1 to 1000000000000, no default | 48000 for audio |
| `resampler` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.45 times the lower of the input and output rates |
| `resampler` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the lower of the input and output rates |
| `resampler` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `resampler` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023; 8192 for 48000 to 44100 in the rational mode |
| `resampler` | `tunable` | - | build | flag, default false | false; true to follow a drifting clock |
| `resampler` | `ppm` | ppm | runtime | f64, -1000 to 1000, default 0 | 0; +100 makes the output rate 100 ppm higher |
| `spectrum` | `fft_size` | - | build | integer, 16 to 65536, default 1024 | 1024; 4096 for finer resolution |
| `spectrum` | `window` | - | build | choice hann / blackman_harris / kaiser, default hann | hann |
| `spectrum` | `kaiser_beta` | - | build | f64, 0 to 50, default 8.6 | 8.6 |
| `spectrum` | `averaging` | - | runtime | integer, 1 to 1024, default 1 | 1; 10 for a smoother display |
| `spectrum` | `interval_samples` | samples | runtime | integer, 1 to 1073741824, no default | default fft_size; fft_size / 2 for 50 % overlap |

<!-- blocks-table:end -->

## The blocks

- **`convert` / `convert_out`:** raw device bytes to `C32`/`CQ15` and back (all interleaved complex formats of [device.md](device.md)). The block works on whole packing groups only (for example 5 bytes for two 10-bit complex samples); `Progress` counts bytes on the raw side and samples on the other. Planar and real layouts cannot be streamed in arbitrary pieces and are rejected, as are 1-bit containers.
- **`scale`:** `gain_db` (runtime) as a real gain, `-120` to `+40` dB for `C32` and `-120` to `0` dB for `CQ15` (0 dB is the largest Q15 value, about -0.0003 dB).
- **`dc_block`:** `r = exp(-2 pi cutoff / fs)` (rounded to Q15 for `CQ15`).
- **`fir`:** a Kaiser-window low-pass from `cutoff_hz`, `transition_hz`, `attenuation_db` (60) and `max_taps` (1023); `cutoff_hz` and `transition_hz` are required.
- **`ddc`:** an NCO at `-frequency_hz` shifts a signal at `+frequency_hz` to 0 Hz, the designed low-pass (default cutoff `0.4 * fs / decimation`, transition `0.1 * fs / decimation`) and the fused `ddc` kernel with a scratch buffer of `max_chunk` samples (default 4096). `frequency_hz` is a runtime parameter and keeps the oscillator phase continuous.
- **`duc`:** the fused `duc` kernel with taps designed at the *output* rate and multiplied by `L`, the NCO at `+frequency_hz` at the output rate; defaults relative to the input rate (cutoff `0.4`, transition `0.1`). `max_chunk` is counted in output samples and must be at least `interpolation`.
- **`fm_demod`:** `quad_demod` + `deemphasis` (the fused `fm_demod` kernel; `deemphasis_us = 0` switches it off) + the audio low-pass with decimation (`resample_real`, L = 1). The discriminator gain is `fs / (2 pi deviation_hz)` for `C32` (the full deviation reads 1.0) and the largest Q15 value for `CQ15` (output unit phase per sample divided by pi; 75 kHz at 240 kHz reads about 0.625). The audio filter has the cutoff `audio_cutoff_hz`, a transition width of a quarter of it and 60 dB; its stop band must lie below the output Nyquist frequency. The block works through its input in pieces of `FM_CHUNK` (256) samples.
- **`fm_mod`:** `fm_mod` with a runtime `deviation_hz`; the phase stays continuous when it changes.
- **`resampler`:** see below.
- **`spectrum`:** see below.

### Resampler

`rate_out_hz` (required, a whole number of Hz) and the input rate define the ratio `L / M`.

- **Rational mode** (`tunable = false`, the default): `L` and `M` are the reduced ratio of the two rates (both at most 1024); the taps are designed at `L * rate_in` (default cutoff `0.45 * min(rate_in, rate_out)`, transition `0.1 * min(..)`, 60 dB) and multiplied by `L`; the block calls `resample` / `resample_real`. The ratio kind is `Constant`. Ratios like 44100/48000 (`147/160`) need thousands of taps at `L * rate_in`: set `max_taps` (the golden case uses 8192). A `ppm` other than 0 needs `tunable = true`; `set_param("ppm")` returns `ReadOnlyParameter`.
- **Tunable mode** (`tunable = true`): a polyphase bank of `P = 128` phases with `K` taps per phase (`K = ceil(taps / 128)` of the `kaiserord` design at `128 * rate_in`) and linear interpolation between neighbouring phases. The prototype low-pass of `128 K` taps is designed at `128 * rate_in` with gain 128; phase `p` uses `h[p + i * 128]`. The read position is a 32.32 fixed-point number of input samples, advanced by `step = round(2^32 * rate_in / rate_out / (1 + ppm * 1e-6))` per output. For an output at position `pos`: `n = pos >> 32`, `frac = pos & 0xFFFF_FFFF`, phase `p = frac >> 25` (top 7 bits), `mu = (frac >> 9) & 0xFFFF` (the next 16 bits, a Q16 weight); the output is `dot(x[n-K+1..=n], bank[p]) * (1 - mu) + dot(.., bank[p+1]) * mu`; the phase 128 is phase 0 applied one sample later (the output then waits for sample `n + 1`). `C32` interpolates in `f32`, `CQ15` computes `round_shift(a (65536 - mu) + b mu, 16)` in `i64` with saturation. `set_param("ppm")` (plus or minus 1000) recomputes the step and applies from the next output. The position is kept relative to the oldest sample still held, so it never overflows however long the stream runs; input between two windows (decimation by more than the work buffer) is skipped. The ratio kind is `Variable`; `timing().ratio` is the nominal `rate_out / rate_in`. With `ppm = +100`, a million input samples at 48 kHz give `1 000 100` outputs (plus or minus 1).

### Spectrum

For every frame the block takes `fft_size` consecutive samples starting at the frame position, multiplies them by a periodic window (`hann`, `blackman_harris` or `kaiser` with `kaiser_beta`), transforms them (`C32`: unnormalised; `CQ15`: block floating point), computes the power per bin, averages the power over `averaging` windows and converts it with `power_db` and the reference `(fft_size * coherent_gain)^2`, so that a full-scale complex tone at a bin centre reads 0 dBFS. The `CQ15` power is *not* `mag_sq` of Q15 (it saturates for large bins and rounds weak bins to zero) but `(re^2 + im^2) 2^(2 exponent - 30)` in `f32` with the exponent of the FFT. The frame is shifted so that the bins run from `-fs/2` to `+fs/2` with DC at index `fft_size / 2`. Consecutive windows start `interval_samples` apart (default `fft_size`; overlapping when smaller, samples are skipped when larger); both `averaging` and `interval_samples` are runtime parameters. `FrameMeta.first_sample` is the input index of the first sample of the first averaged window, counted since construction or the last `reset`.

## Tests

- Golden cases (`design/`, `dot/`, `blocks/`): blocks are driven in one piece and in random chunks; results must be identical and match SciPy/NumPy (`ddc` 130 dB, FM demodulator 133 dB, rational resampler 139 dB, tuned resampler 140 dB (`C32`) and bit-exact (`CQ15`), spectrum within 0.0002 dB (`C32`) and 0.0013 dB (`CQ15`)).
- `tests/describe.rs`, `construction.rs`, `params.rs`, `behaviour.rs`, `timing.rs`, `ppm.rs`, `functional.rs`, `design.rs`, `storage.rs` in `crates/blocks/sdrk-blocks`: the rules above, exact storage, parameters, chunk invariance and `reset` of every block and sample type, extreme inputs (silence, full scale, NaN, infinity), impulse delays, output counts, pass bands and stop bands, DUC-DDC and FM modulator-demodulator loops, spectrum levels.
- Suite cases (no heap, `ArenaStorage` on the suite arena): `block_ddc_vs_chain`, `block_fm_demod_sinad` (the receive chain of M08 rebuilt from blocks, thresholds 40/35 dB; it reaches about 98/86 dB because the designed filters are better than the Hann filters of the kernel chain), `block_spectrum_tone_level` (0 dBFS plus or minus 0.05 dB for `C32`, plus or minus 0.3 dB for `CQ15`), `block_resampler_ppm_count`.
