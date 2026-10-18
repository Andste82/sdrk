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

`WriteReporter` prints it to any `core::fmt::Write`, `StdReporter` (feature `std`) to stdout. Registered by M03: `rng_known_answer`, `gen_tone_frequency`, `encode_decode_roundtrip_all_formats`, `scale_fallback_vs_reference` (quick) and `scale_block_invariance` (full). Registered by M05: `convert_all_formats_vs_encoder`, `rotate_c32_vs_reference`, `rotate_cq15_vs_reference`, `rotate_block_invariance`, `fir_c32_vs_reference`, `fir_cq15_vs_reference`, `fir_block_invariance`, `fir_impulse_response` (quick) and `extremes_convert_rotate_fir` (full). They compare `Fallback` and the facade with `Reference` and need no files, so they run on microcontrollers too. New kernels add their cases to `cases.rs`. On the host, `cargo test -p sdrk-testkit --features std` runs the whole suite in `tests/suite.rs`.

## Golden vectors

Golden vectors are produced by `tests/golden/generate.py` with NumPy/SciPy (pinned exactly) and an independent Python implementation of the Q15 rules. They are committed to `tests/golden/vectors/<kernel>/<case>.json` plus binary arrays and loaded by `sdrk_testkit::golden`. Each case stores its tolerance (`bit_exact`, `tolerance` with a minimum SNR, or `max_abs_err`), so tolerances never live in Rust code. See [`tests/golden/README.md`](../tests/golden/README.md) for the format, regeneration (`python tests/golden/generate.py --check`) and how to add a case. Present cases: `q15_ops/{from_float_4096, mul_pairs_4096, cmul_pairs_4096}`, `scale/{c32_random_4096, cq15_random_4096}`, `convert/<format>_256` for every format constant plus `cu8_swapiq_256` and `cs16_bigendian_256` (arrays `input` bytes, `expected_f32`, `expected_q15`; all bit-exact), `rotate/{c32_noise_f0p1234_8192, c32_negfreq_8192, c32_wrap_4096, cq15_noise_f0p05_8192, cq15_fullscale_tone_4096}` and `fir/{lowpass63_c32_8192, lowpass63_cq15_8192, long255_c32_16384, single_tap_c32_1024, gain_overflow_cq15_2048}`. The Rust golden tests of the stateful kernels feed the input once in one piece and once in random chunks.
