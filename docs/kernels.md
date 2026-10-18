# Kernels

How kernels are structured in SDRk, and the numeric contract of each one.

## The one-trait-per-kernel pattern

For every kernel `foo` there is:

1. A trait in `sdrk_core::kernels::<category>`, generic over the sample type:
   `trait Foo<T: Sample> { fn foo(&self, [state,] input: &[T], output: &mut [T], [params]) -> Result<()>; }`.
   Rust has no stable specialization, so the trait itself carries the sample type.
   - Same-rate kernels require `output.len() == input.len()`, otherwise `Error::LengthMismatch { expected: input.len(), got: output.len() }`.
   - Rate-changing kernels return `Result<Progress>`; they consume as much input as possible and produce at most `output.len()` samples. Unconsumed input is resubmitted by the caller.
   - Stateful kernels take `state: &mut FooState<..>` right after `&self`.
2. `Fallback` (`sdrk_core::Fallback`): the portable, optimised-scalar implementation for every supported sample type.
3. `Reference` (`sdrk_core::Reference`, feature `reference`): the executable specification, f64 for f32 kernels and the exact integer model for Q15 kernels. Clarity over speed.
4. The facade `sdrk::Sdrk` implements the trait for every sample type by delegation, written with `sdrk::delegate!`:

   ```rust,ignore
   delegate!(Scale<C32>::scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> => Fallback);
   ```

   A backend changes only the `=> Backend` target (guarded by `#[cfg(..)]`). The same entries build the table behind `sdrk::backend_for(kernel, sample_type)`, which returns the implementation name or `"none"`.
5. A row in the contract table below.

`sdrk_core::map_kernel!` generates the body of a same-rate, stateless, per-sample fallback: it checks the lengths and applies a closure to every sample. It is the starting point for later SIMD "process one" helpers.

## Numeric contract

Classes: *bit-exact* (fallback equals the reference bit for bit) and *tolerance* (minimum SNR against the f64 reference). Q15 rules: round half up, saturate, `-1 * -1 = 32767`.

| Kernel | Trait | Sample types | Class | Contract |
| --- | --- | --- | --- | --- |
| `scale` | `kernels::basic::Scale` | `C32` | tolerance | SNR >= 140 dB vs. the f64 reference; plain f32 multiply per component, no FMA. The f64 path and the f32 path can differ in the last bit, so bit-exactness is not promised. |
| `scale` | `kernels::basic::Scale` | `CQ15` | bit-exact | `q15_mul` per component with the Q15 gain |
| `convert` | `kernels::convert::Convert` | `C32`, `CQ15` | bit-exact | exact mapping rules below; fallback equals the reference bit for bit |
| `convert_real` | `kernels::convert::ConvertReal` | `f32`, `Q15` | bit-exact | same rules for `Layout::Real` |
| `rotate` | `kernels::mix::Rotate` | `C32` | tolerance | SNR >= 120 dB vs. the golden f64 expected value; table interpolation in f32 without FMA |
| `rotate` | `kernels::mix::Rotate` | `CQ15` | tolerance | SNR >= 80 dB vs. the golden f64 expected value; Q15 table, `i64` interpolation, `cq15_mul` |
| `fir` | `kernels::filter::Fir` | `C32` | tolerance | SNR >= 110 dB vs. the f64 reference; f32 accumulation, taps ascending, no FMA |
| `fir` | `kernels::filter::Fir` | `CQ15` | bit-exact | Q30 products summed in `i64`, one rounding and saturation per component |
| `fir_decimate` | `kernels::decim::FirDecimate` | `C32` | tolerance | SNR >= 110 dB vs. the f64 reference; accumulation as `fir` (f32 ascending, no FMA); only the emitted outputs are computed |
| `fir_decimate` | `kernels::decim::FirDecimate` | `CQ15` | bit-exact | `Acc64` per component, one rounding and saturation per emitted output |
| `resample` | `kernels::decim::Resample` | `C32` | tolerance | SNR >= 110 dB (100 dB for L/M = 147/160 with 1470 taps) vs. `scipy.signal.upfirdn`; polyphase, f32 ascending, no FMA |
| `resample` | `kernels::decim::Resample` | `CQ15` | bit-exact | `Acc64` per component over the polyphase branch |
| `resample_real` | `kernels::decim::ResampleReal` | `f32` | tolerance | same as complex `resample` |
| `resample_real` | `kernels::decim::ResampleReal` | `Q15` | bit-exact | same as complex `resample` |
| `quad_demod` | `kernels::demod::QuadDemod` | `C32` | tolerance | SNR >= 100 dB vs. the f64 reference; f32 product without FMA, `libm::atan2f` |
| `quad_demod` | `kernels::demod::QuadDemod` | `CQ15` | tolerance | SNR >= 70 dB vs. the f64 reference (at most one LSB from the reference for any input); exact `i64` product, `libm::atan2` in f64, `q15_mul` with the gain |
| `deemphasis` | `kernels::demod::Deemphasis` | `f32` | tolerance | SNR >= 100 dB; `y += alpha * (x - y)` without FMA, then `flush_denormal` |
| `deemphasis` | `kernels::demod::Deemphasis` | `Q15` | bit-exact | `y = sat16(y + round_shift(alpha * (x - y), 15))` in `i64` |
| `dc_block` | `kernels::demod::DcBlock` | `C32` | tolerance | SNR >= 100 dB; `y = (x - x_prev) + r * y_prev` per component, state flushed |
| `dc_block` | `kernels::demod::DcBlock` | `CQ15` | bit-exact | `y = sat16((x - x_prev) + round_shift(r * y_prev, 15))` per component |
| `window` | `kernels::spectrum::Window` | `C32` | bit-exact | f32 multiply per component (equal to the f64 product rounded once) |
| `window` | `kernels::spectrum::Window` | `CQ15` | bit-exact | `q15_mul` per component |
| `mag_sq` | `kernels::spectrum::MagSq` | `C32` | tolerance | SNR >= 140 dB; `re * re + im * im` without FMA |
| `mag_sq` | `kernels::spectrum::MagSq` | `CQ15` | bit-exact | `sat16(round_shift(re^2 + im^2, 15))`; full scale saturates |
| `power_db` | `kernels::spectrum::PowerDb` | `f32`, `Q15` | tolerance | largest absolute error <= 1e-4 dB; `10 * log10f(max(p, 1e-20)) + offset` |

### `convert` and `convert_real`

`Convert<T>::convert(&format, input_bytes, output) -> Result<usize>` decodes whole samples and returns their number; `ConvertReal<T>` does the same for `Layout::Real`. The effective integer value `e` of each component follows `docs/device.md` (byte order, alignment, sign extension, packing, I/Q swap, planar halves). Errors: a trailing partial sample or packing group gives `InvalidParameter("partial sample")`, an output shorter than `format.samples_in_bytes(input.len())` gives `BufferTooSmall`, `Bits1` gives `Unsupported`, a layout that does not match the trait gives `InvalidParameter`. A longer output is allowed; only the decoded prefix is written.

- **To `f32`/`C32`:** `(e - offset) / full_scale` in f32 with `offset = 127.5` for `U8` and 0 otherwise. For at most 24 effective bits (and an explicit full scale up to 2^24) every operation is exact or a single correctly rounded division. Beyond that (`I32` with more than 24 bits) the value is computed in f64 and rounded once to f32. `F32` containers are copied (byte-swapped when big-endian); NaN, infinities and -0.0 pass through unchanged.
- **To `Q15`/`CQ15`:** `U8`: `q = (2u - 255) << (15 - bits)`, i.e. `(u - 127.5) * 256` for 8 bits; signed power-of-two full scale with at most 16 bits: `q = e << (16 - bits)`; more than 16 bits: `sat16(round_shift(e, bits - 16))`; explicit full scale `n`: `sat16(floor((e * 65536 + n) / (2n)))` (`U8` uses `(2u - 255)` instead of `2e` and `32768` instead of `65536`), which is round half up of `e * 32768 / n`; `F32`: `q15_from_f32` (NaN gives 0, infinities saturate).
- **Packed containers** ignore `align` and `endian` (they are LSB-first bit streams); `Msb` alignment applies to `U8`, `I8`, `I16` and `I32`.

`view_cq15` and `view_c32` return a zero-copy view when the bytes already have the layout of the internal type (little-endian, interleaved, no I/Q swap, power-of-two full scale for `CS16`) and are aligned and whole; the view equals what `convert` would write, bit for bit. Empty input gives an empty view.

### `rotate`

`Rotate<T>::rotate(&mut Nco, input, output)` multiplies by `exp(j 2 pi phase_n / 2^32)`; the phase model is in `docs/timing.md`. The tables are generated by `crates/core/sdrk-core/build.rs` (build dependency `libm`): `COS_F32[k] = cos(2 pi k / 4096)` as f32 and `COS_Q15[k]` the same f64 value rounded by the Q15 rule, `k = 0..=4096`. Sine is cosine at `phase - 2^30`. Both tables are plain immutable data.

- **Fallback `C32`:** `idx = phase >> 20`, `f = (phase & 0xFFFFF) / 2^20` in f32, `c = a + (b - a) * f` (no FMA), result `(re c - im s, re s + im c)`.
- **Fallback `CQ15`:** `f15 = (phase >> 5) & 0x7FFF`, `c = sat16(a + round_shift((b - a) * f15, 15))` in `i64`, then `cq15_mul`.
- **Reference:** `theta = 2 pi phase / 2^32` in f64, `libm::cos/sin`, f64 complex product, rounded once.
- The table value at angle 0 is `32767/32768` (not 1.0), so `CQ15` rotation by zero is the identity only for `|x| <= 16384` and within one LSB for larger values. In `C32` the sine of angle 0 is the table entry for 3 pi / 2 (about -1.8e-16), so rotation by zero is exact whenever `|im| * 1.8e-16` is below half an ULP of `re` (always true for ordinary signals; an input with `re = 0` and `im != 0` receives a residue of about 1e-16).
- A rotated `CQ15` sample whose magnitude exceeds 1 saturates one component; the golden cases clip their expected value the same way.

### `fir`

`FirState<T, Taps, Hist>` owns real taps (`T::Real`, so `f32` for `C32` and `Q15` for `CQ15`) and a history of the last `taps.len() - 1` samples (oldest first), both in caller-provided storage (`[T; N]`, `&mut [T]`, `Vec<T>`). `FirState::new` rejects empty taps (`InvalidParameter`) and a history of the wrong length (`LengthMismatch`) and zeroes the history; `reset()` clears it. `Fir<T>::fir` computes `y[n] = sum h[k] x[n - k]` with samples before the start taken from the history and updates the history to the last `N - 1` samples of (old history ++ input). A call with empty slices changes nothing; unequal lengths give `LengthMismatch` and leave the state untouched.

- **`C32` fallback:** separate f32 accumulators for `re` and `im`, starting at 0.0, `k` ascending, no FMA. The order is fixed so the result does not depend on the block size.
- **`CQ15` fallback and reference:** `Acc64` per component, `sat16(round_shift(acc, 15))` once. Bit-exact.
- **`C32` reference:** f64 accumulation, one cast to f32.

### `fir_decimate`

`FirDecimState<T, Taps, Hist>` holds the real taps, a history of `N - 1` samples, the factor `M >= 1` (`new(taps, history, factor)`, `reset()`, getters `factor()`, `taps()`, `history()`, `countdown()`) and a countdown of input samples to skip. With `y_full[n] = sum h[k] x[n-k]` over the whole stream (zero history at the start and after `reset`) the outputs are `y[m] = y_full[m * M]`: the first input sample after a reset produces the first output, and the `m`-th output follows `M` input samples after the previous one. The fallback computes only the outputs it emits. `fir_decimate` returns `Progress`: an input sample that would produce an output is not consumed when the output buffer is full, so `consumed` stops right before it; samples that produce nothing are consumed up to that point. The caller resubmits `input[consumed..]`. The accumulation is the one of `fir`, so `M = 1` equals `fir` bit for bit. Timing: ratio `1/M`, group delay `Delay::new(N - 1, 2)` in input samples.

### `resample` and `resample_real`

`ResampleState<T, Taps, Hist>` (complex signals, `ResampleRealState` for `f32`/`Q15` signals) holds the taps `h` designed at the upsampled rate (the interpolation gain `L` is part of the taps, the caller includes it), `up = L >= 1`, `down = M >= 1`, a history of `ceil(N / L)` input samples (`new` checks the length, `LengthMismatch` otherwise) and the output phase. The definition is `scipy.signal.upfirdn(h, x, up=L, down=M)` restricted to the outputs that depend only on received input: `y[j] = sum_k h[k] u[j M - k]` with `u[i] = x[i / L]` when `i mod L == 0` and 0 otherwise. Output `j` is produced as soon as input sample `floor(j M / L)` has been received; evaluated as a polyphase filter with `t = j M`, `p = t mod L`, `b = t div L`: `y[j] = sum_{i >= 0, p + i L < N} h[p + i L] x[b - i]` with zeros before the first sample.

The number of outputs for `n` input samples after a reset is `K = floor((n L - 1) / M) + 1` (check: `L = 3`, `M = 2`, `n = 1` gives `K = 2`). The state keeps the output phase as `q = j M - L c` (`j` outputs produced, `c` input samples consumed): an output is ready while `q < 0`, which is when `q` lies in `[-L, -1]` and the polyphase branch is `p = q + L`; a producing step adds `M`, a consuming step subtracts `L`. A sample is consumed only while no output is pending, so the newest sample is always the one the next output refers to and the history of `ceil(N / L)` samples suffices. The phase stays bounded however long the stream runs. `Progress` semantics are those of `fir_decimate`. Timing: ratio `L / M`, group delay `Delay::new(N - 1, 2 L)` in input samples. With `L = M = 1` the resampler equals `fir`.

The Q15 variants cannot hold a tap gain of `L` above 1; the golden case `resample/l3m2_n96_cq15_6000` therefore scales the taps by `1/3` (a documented gain loss of 3 that the caller restores after the resampler).

### `quad_demod`

`QuadDemodState<T>` keeps the previous sample (zero after `new`/`reset`). `output[n] = gain * arg(x[n] * conj(x[n-1]))` with `arg(0) = 0`; the first output after a reset is therefore 0. `C32`: the product in f32 without FMA, `libm::atan2f(im, re) * gain`, radians times the gain. `CQ15`: the product in exact `i64` (`re = ar*br + ai*bi`, `im = ai*br - ar*bi` with `a = x[n]`, `b = x[n-1]`), `phi = libm::atan2(im, re) / pi` in f64 rounded by the Q15 rule (unit: phase / pi, so a half turn is 32767), then `q15_mul(phi, gain)`; an accelerated variant may use CORDIC later. The reference does everything in f64 with a single rounding at the end; the Q15 fallback may differ from it by one LSB because it rounds twice. A typical gain for FM is `fs / (2 pi deviation)` (f32) so that full deviation maps to +-1. Timing: ratio `1/1`, group delay `Delay::new(1, 2)`.

### `deemphasis`

`DeemphasisState<T>::new(tau_s, rate)` computes `alpha = 1 - exp(-1 / (fs * tau))` in f64 and rounds it to the sample type (cast for `f32`, the Q15 rule for `Q15`); `tau_s` must be finite and positive. `f32`: `y = y + alpha * (x - y)` without FMA, then `flush_denormal`. `Q15`: `d = x - y` in `i64`, `y = sat16(y + round_shift(alpha * d, 15))`. After `tau * fs` samples the step response reaches 63.2 percent (within one sample). The filter is an IIR filter, so its delay depends on the frequency; `group_delay` reports 0 and the delay is not part of exact event timing. A NaN or infinite `f32` input poisons the state until `reset()` (a finite state is guaranteed only for finite input).

### `dc_block`

`DcBlockState<T>::new(r)` with `0 < r < 1` (Q15: raw 1 to 32767) holds `r`, the previous input and the previous output. Per component `y[n] = x[n] - x[n-1] + r * y[n-1]`. `f32`: as written without FMA; the stored `y_prev` is flushed with `flush_denormal`, the returned sample is not. `Q15`: `y = sat16((x - x_prev) + round_shift(r * y_prev, 15))`. **Limit of the Q15 variant:** the round-half-up feedback has a dead band, so a constant input does not decay to exactly zero but settles within `floor(0.5 / (1 - r_q15))` LSB per component; for `r_q15 = 32735` (0.999) that is 496 LSB (magnitude below 0.022). The `f32` variant removes a constant to below 1e-3 within 10 000 samples at `r = 0.999`. Timing: ratio `1/1`, group delay 0.

### Windows, `window`, `mag_sq`, `power_db`

`sdrk_core::window` writes windows into caller slices of `f32` or `Q15` (computed in f64 with `libm`): `hann`, `blackman_harris` (4-term, `a0 = 0.35875`, `a1 = 0.48829`, `a2 = 0.14128`, `a3 = 0.01168`), `kaiser(out, beta, sym)` (`beta` finite and not negative), `bessel_i0` (power series, at most 500 terms) and `window_gain(&[f32]) -> (coherent gain, noise-equivalent bandwidth)`. `Symmetry::Symmetric` uses the denominator `N - 1`, `Symmetry::Periodic` the symmetric window of length `N + 1` without its last value (`scipy.signal.get_window(..., fftbins=True)`); a window of length 1 is `[1.0]`, an empty slice is accepted. A symmetric window is symmetric bit for bit. The periodic Hann window has `(0.5, 1.5)` as gain and bandwidth. In `Q15` the peak 1.0 saturates to 32767.

`window(input, output, w)` multiplies per component (`w.len() == input.len()`): `f32` as a plain f32 multiply (equal to the f64 product rounded once because the product of two f32 values is exact in f64), `Q15` with `q15_mul`. `mag_sq` computes `re * re + im * im` (`f32`, no FMA) or `sat16(round_shift(re^2 + im^2, 15))` (`Q15`, full-scale samples saturate to 32767). `power_db(input, output_f32, offset_db)` computes `10 * log10(max(p, 1e-20)) + offset_db` with `libm::log10f` on the f32 power (`Q15` input is `raw / 32768`; the reference uses `libm::log10` in f64). Negative, zero and NaN input count as `1e-20` (so -200 dB plus the offset), and the kernel never panics.

## How kernels are tested

Every kernel is covered by three layers, all provided by `sdrk-testkit`:

1. **Golden vectors** (M04) and the **reference**: the fallback is compared with `Reference` (bit-exact kernels with `bit_exact`, tolerance kernels with `compare_complex` and the SNR from the contract table).
2. **Block-size invariance** (R6.2): stateful kernels and blocks must produce identical bits however the input is split. `check_block_invariance(input, out_whole, out_chunked, ChunkPlan::new(seed, max), make_state, step)` runs `step` once over the whole input and once over random chunk sizes (deterministic PCG32) and reports the first differing output index. `step` may consume less than it is offered; the harness resubmits the rest. Stateless kernels such as `scale` pass it trivially and are still registered, so the harness itself stays covered.
3. **Extreme inputs**: `r#gen::extreme::silence`, `full_scale_alternating` (`+-(1 - 2^-15)` on both components) and `f32_specials` (0.0, -0.0, NaN, +inf, -inf, a denormal, `f32::MAX`) feed every kernel. A kernel must not panic, must saturate Q15 results and must keep feedback state finite (`flush_denormal`).

Test inputs come from deterministic generators (`r#gen::tone`, `noise`, `chirp`, ...) that depend only on the absolute sample index, so chunked and whole generation agree bit for bit. Raw device formats are checked against the independent oracle `encode::encode` / `encode::decode_reference`, never against the code under test.

### The suite

`suite::all_cases()` lists every registered `TestCase { name, tier, run }`. A case is a plain function over an `Arena` (no heap, no `cargo test`), so the same suite runs on the PC and inside emulators. `run_suite(cases, tier, arena, reporter)` resets the arena before each case; `Tier::Quick` runs the quick cases (64 KiB arena), `Tier::Full` all of them (256 KiB). Reporters print the protocol parsed by the emulator runners:

```text
SDRK-SUITE START <n>
SDRK-CASE PASS <name>
SDRK-CASE FAIL <name> <msg> <detail>
SDRK-SUITE END passed=<p> failed=<f>
```

`WriteReporter` prints it to any `core::fmt::Write`, `StdReporter` (feature `std`) to stdout. Registered by M03: `rng_known_answer`, `gen_tone_frequency`, `encode_decode_roundtrip_all_formats`, `scale_fallback_vs_reference` (quick) and `scale_block_invariance` (full). Registered by M05: `convert_all_formats_vs_encoder`, `rotate_c32_vs_reference`, `rotate_cq15_vs_reference`, `rotate_block_invariance`, `fir_c32_vs_reference`, `fir_cq15_vs_reference`, `fir_block_invariance`, `fir_impulse_response` (quick) and `extremes_convert_rotate_fir` (full). Registered by M06 (quick unless noted): `fir_decimate_c32_vs_reference`, `fir_decimate_cq15_vs_reference`, `fir_decimate_block_invariance`, `resample_c32_vs_reference`, `resample_cq15_vs_reference`, `resample_real_vs_reference` (each over the ratios 1/5, 3/2 and 147/160 where it applies), `resample_block_invariance`, `quad_demod_c32_vs_reference`, `quad_demod_cq15_vs_reference`, `quad_demod_block_invariance`, `deemphasis_f32_vs_reference`, `deemphasis_q15_vs_reference`, `deemphasis_block_invariance`, `dc_block_c32_vs_reference`, `dc_block_cq15_vs_reference`, `dc_block_block_invariance`, `window_vs_reference`, `window_generators`, `mag_sq_vs_reference`, `power_db_vs_reference` and `extremes_rx_kernels` (full). They compare `Fallback` and the facade with `Reference` and need no files, so they run on microcontrollers too. New kernels add their cases to `cases.rs`. On the host, `cargo test -p sdrk-testkit --features std` runs the whole suite in `tests/suite.rs`.

## Golden vectors

Golden vectors are produced by `tests/golden/generate.py` with NumPy/SciPy (pinned exactly) and an independent Python implementation of the Q15 rules. They are committed to `tests/golden/vectors/<kernel>/<case>.json` plus binary arrays and loaded by `sdrk_testkit::golden`. Each case stores its tolerance (`bit_exact`, `tolerance` with a minimum SNR, or `max_abs_err`), so tolerances never live in Rust code. See [`tests/golden/README.md`](../tests/golden/README.md) for the format, regeneration (`python tests/golden/generate.py --check`) and how to add a case. Present cases: `q15_ops/{from_float_4096, mul_pairs_4096, cmul_pairs_4096}`, `scale/{c32_random_4096, cq15_random_4096}`, `convert/<format>_256` for every format constant plus `cu8_swapiq_256` and `cs16_bigendian_256` (arrays `input` bytes, `expected_f32`, `expected_q15`; all bit-exact), `rotate/{c32_noise_f0p1234_8192, c32_negfreq_8192, c32_wrap_4096, cq15_noise_f0p05_8192, cq15_fullscale_tone_4096}` and `fir/{lowpass63_c32_8192, lowpass63_cq15_8192, long255_c32_16384, single_tap_c32_1024, gain_overflow_cq15_2048}`. M06 adds `fir_decimate/{m10_n63_c32_20000, m10_n63_cq15_20000, m3_n31_c32_9000}`, `resample/{l1m5_n121_c32_10000, l3m2_n96_c32_6000, l147m160_n1470_c32_16000, l3m2_n96_cq15_6000}`, `resample_real/l1m5_n121_f32_10000`, `quad_demod/{fm_tone_c32_20000, fm_tone_cq15_20000}`, `deemphasis/{tau50us_fs48k_f32_8192, tau50us_fs48k_q15_8192}`, `dc_block/{r0p999_c32_8192, r0p999_cq15_8192}`, `window/{hann_periodic_1024, hann_symmetric_1023, blackmanharris_periodic_4096, kaiser8_symmetric_255, apply_hann_c32_1024, apply_hann_cq15_1024}`, `mag_sq/{c32_4096, cq15_4096}` and `power_db/f32_4096`. The expected resampler output is `upfirdn(h, x, L, M)[:K]` with `K = floor((n * L - 1) / M) + 1` (written in `cases/resample.py`). The Rust golden tests of the stateful kernels feed the input once in one piece and once in random chunks.
