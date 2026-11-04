# Blocks

`sdrk-blocks` (layer `crates/blocks`, `no_std`, no heap) turns kernels into blocks that think in parameters with units ("DDC to 600 kHz, decimate by 10, cutoff 100 kHz") and describe themselves. A block translates its parameters into kernel states, owns no memory of its own and calls the facade `sdrk::Sdrk` for every kernel, so backends speed up the blocks automatically. The pipeline ([pipeline.md](pipeline.md)) builds blocks from JSON with the same names and ranges that are written down here.

## The block interface

```rust,ignore
pub trait Block: Send {
    type In: Ports;
    type Out: Ports;
    fn description(&self) -> &'static BlockDescription;
    fn process(&mut self, input: <Self::In as Ports>::Ref<'_>,
               output: <Self::Out as Ports>::Mut<'_>) -> Result<Progress>;
    fn timing(&self) -> BlockTiming;               // { ratio, kind, delay }
    fn reset(&mut self);
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
    fn param(&self, name: &str) -> Result<ParamValue>;
    fn min_input(&self) -> usize;
    fn map_index(&self, rel_in: u64) -> u64;       // default: ceil(rel_in * L / M)
    fn frame_len(&self) -> usize { 0 }             // values per frame for Out = Frame
}
    fn reset(&mut self);
    fn set_param(&mut self, name: &str, value: ParamValue) -> Result<()>;
    fn param(&self, name: &str) -> Result<ParamValue>;
    fn min_input(&self) -> usize;
    fn map_index(&self, rel_in: u64) -> u64;       // default: ceil(rel_in * L / M)
}
```

- The input and the output of a block are *port types* (`Ports`): a stream of one sample type (`C32`, `CQ15`, `f32`, `Q15`, `u8`; the sealed trait `StreamElem`), a pair of streams `Two<A, B>` or frames `Frame`. A stream element is its own port type with `Ref = &[T]` and `Mut = &mut [T]`, so a stream block keeps its signature: `type In = C32;` and `fn process(&mut self, input: &[C32], output: &mut [C32])`. A pair block reads `(&[A], &[B])` and writes `(&mut [A], &mut [B])`; a frame block writes a `FrameOut { values, meta }`.
- Lockstep rule for `Two`: both input slices of a call have the same length and the block consumes the same number of samples from each (`Progress.consumed` counts pairs); both output slices have the same length and the block writes the same number of samples to each (`Progress.produced` counts pairs). Unequal lengths are `LengthMismatch` and change nothing.
- `process` follows the contract of the rate-changing kernels: it consumes as much input as it can and produces at most `output.len()` samples; the caller resubmits `input[consumed..]`. A call that can neither consume nor produce returns `Progress { 0, 0 }` (no input, no output room, or less than one packing group). A resampler that ran out of output room keeps the remaining outputs pending and delivers them in the next call, also with an empty input.
- The output depends only on the absolute input position, never on how the input was cut into calls or how big the output buffer was (tested for every block with random chunk plans, including 1-sample chunks).
- `map_index(rel_in)` maps a count of input samples since the last origin (construction or `reset`) to the number of output samples that include them: the first output sample that includes input sample `n` is `map_index(n)`. The default is `ceil(rel_in * L / M)` with the exact ratio of `timing()`; the tunable resampler overrides it with its position step (`ceil(rel_in * 2^32 / step)` in `u128`). The pipeline uses it for the index domains of [timing.md](timing.md#index-domains-in-the-pipeline).
- `timing()` is exact: `ratio` is output rate divided by input rate (`Ratio`), `kind` is `Constant` or `Variable` (the tunable resampler), `delay` the group delay in input samples as an exact fraction (see [timing.md](timing.md)).
- `reset()` clears all history (the pipeline calls it on gaps) and keeps the parameters; the output after a reset equals that of a fresh block.
- `set_param` changes a *runtime* parameter from the first sample of the next `process` call. Errors: `UnknownParameter` (name not in the description), `ReadOnlyParameter` (a build parameter), `WrongParamType`, `OutOfRange { name, min, max, got }` (also for NaN). A rejected value changes nothing. `param` returns the value in use, also for build parameters and for defaults that depend on the sample rate (`cutoff_hz` of a `ddc` reads `0.4 * fs / decimation`).
- Blocks that output frames instead of a stream (the spectrum) are `Block<Out = Frame>`: `process(input, FrameOut { values, meta })` completes at most one frame per call and stops consuming right after it. It writes the frame into `values` (`values.len() != frame_len()` is `LengthMismatch`), sets `*meta = Some(FrameMeta { first_sample, averaged, exponent })` and returns `produced: 1`; without a completed frame `produced` is 0 and `*meta` is left unchanged (the caller sets it to `None` before the call). `exponent` is always 0: the block applies the exponent of the block-floating-point FFT itself. The `timing()` of the spectrum is the nominal ratio `1 / interval_samples` (`Variable`) with the delay `fft_size - 1`; the pipeline does not use it for rates.

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

`validate` applies the limits that depend on the input rate (`frequency_hz` within plus or minus half the rate, a stop band below half the rate). Parameters with a rate-dependent default (`cutoff_hz` and `transition_hz` of `ddc`, `duc` and `resample`) are `Option<f64>` in the config: `None` selects the default. `FooConfig::get` of an unset one is an error; the block reports the resolved value. `set` and `get` work on build and runtime parameters and check kind and range with the same table (`ParamDesc::check`) that the description publishes.

Blocks exist for the complex sample types `C32` and `CQ15` (`ConvertBlock<T, S>`, `ConvertOutBlock`, `ScaleBlock`, `DcBlockBlock`, `FirBlock`, `DdcBlock`, `DucBlock`, `FmDemodBlock` (`C32`/`CQ15` in, `f32`/`Q15` out), `FmModBlock` (`f32`/`Q15` in, `C32`/`CQ15` out), `ResampleBlock`, `SpectrumBlock`, `SquelchBlock`, `AgcBlock`, `AmDemodBlock` (`C32`/`CQ15` in, `f32`/`Q15` out)) and for the real types in `ResampleRealBlock<R, S>` (`f32`, `Q15`), which is the resampler for audio.

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

## Chains

`Chain<A, B, S>` (M27, module `chain.rs`) is two blocks in a row, itself a `Block`, so chains nest. It needs no pipeline, no JSON and no heap: the intermediate buffer (two for a pair of streams) comes from the same `Storage` as the blocks. It is for programs that know their chain at compile time, a microcontroller receiver above all.

```rust
use sdrk::{Rate, C32};
use sdrk_blocks::{
    chain, AlignedBytes, ArenaStorage, Block, ConvertBlock, ConvertConfig, DdcBlock, DdcConfig,
    FmDemodBlock, FmDemodConfig,
};

fn receiver() -> Result<(), Box<dyn std::error::Error>> {
    let rate = Rate::hz(2_400_000)?;
    let mut mem = AlignedBytes([0u8; 65536]);
    let mut arena = ArenaStorage::new(&mut mem.0);
    let ddc_cfg = DdcConfig { frequency_hz: 600e3, decimation: 10, ..DdcConfig::default() };
    let mut rx = chain!(
        &mut arena, 256;
        ConvertBlock::<C32, _>::new(&ConvertConfig::default(), rate, &mut arena)?,
        DdcBlock::<C32, _>::new(&ddc_cfg, rate, &mut arena)?,
        FmDemodBlock::<C32, _>::new(&FmDemodConfig::default(), Rate::hz(240_000)?, &mut arena)?
    );
    rx.set_param("1.0.frequency_hz", sdrk_blocks::ParamValue::F64(610e3))?;
    Ok(())
}
```

- **Types:** `A::Out` must be a `ChainItem` (sealed): a stream element (`StreamElem`, one intermediate buffer) or, since M31, a pair of streams `Two<X, Y>` of stream elements (two buffers of `chunk` samples each, written by `A` and consumed by `B` in lockstep, M26 §1). A chain cannot pass `Frame` between its stages, and a port type that is not a stream element (the packets of M37) does not compile. `A::In` and `B::Out` may be any port type, so a chain can end in a frame block (`convert -> spectrum`) or start with a pair of streams. `S` is the storage of the buffer; it is the storage type of the blocks in the usual case.
- **`process`:** the buffer holds what `A` produced and `B` has not consumed. One call feeds `B` while the buffer holds `b.min_input()` samples (or the input is used up and the buffer is not empty), moves the rest to the front, fills the buffer from `A`, and repeats until the output is full or neither stage progresses. `Progress.consumed` is what `A` consumed, `Progress.produced` what `B` produced. Each stage is called at most `2 + input.len() / chunk + output.len()` times per call, never per sample. The result is the same as running `A` over the whole input and then `B` over the whole result, for any cut of input and output, as far as the stages are invariant themselves (the `f32` `fm_demod` is invariant up to the tolerance of its SIMD kernels, the Q15 blocks to the bit). The buffer is raised to `b.min_input()` and at least 1. A chain does not check the lengths of a call itself: a pair of unequal streams or a wrong frame length is reported by the stage that sees it.
- **Timing:** `ratio = ratio_a * ratio_b`; `kind` is `Variable` if either is; `delay = delay_a + delay_b / ratio_a` in input samples of `A`, exact (`Delay::add`, `Delay::div_ratio`); `min_input() = max(a.min_input(), ceil(b.min_input() / ratio_a))`; `map_index(n) = b.map_index(a.map_index(n))`; `frame_len()` is that of `B`; `reset()` resets both stages and empties the buffer.
- **Parameters:** `"<stage>.<name>"`, stage `0` for `A`, `1` for `B`; nested chains add prefixes. `chain!(storage, chunk; a, b, c)` nests to the right, so the names read `0.`, `1.0.` and `1.1.` (`"1.0.frequency_hz"` above). A name without prefix or with another stage is `UnknownParameter`; the error of a stage is returned unchanged.
- **`chain!`:** two to eight blocks, one `chunk` for all buffers. The macro expands to `Chain::new(...)?`: the calling function returns a `Result` whose error type is built from `sdrk_blocks::Error`.
- **Storage:** `Chain::storage_bytes(chunk)` is `chunk * size_of::<A::Out>()` for a stream; for a pair `Two<X, Y>` it counts both buffers, `chunk * size_of::<X>()`, the padding that aligns the second buffer behind the first, and `chunk * size_of::<Y>()`. It is exactly what `new` takes from an arena that starts at an aligned address (`AlignedBytes`) and is still empty. At an unaligned position up to `align_of - 1` bytes of padding per buffer come on top. See [embedded.md](embedded.md#sizing-the-arena-of-a-chain).
- **Description:** `CHAIN_DESCRIPTION` has the kind `chain` and empty port lists (they follow from the stages and cannot be written down statically). A chain is a Rust construct: it is not a pipeline kind and not in `all_descriptions()`, and `cargo xtask check` (names) lists `chain` and `fork` (M31) as the exceptions to the rule that a kind lives in `blocks/<kind>.rs` with a `<Kind>Block`.

## Composite blocks

A *composite block* is a block whose result is exactly that of a small graph of primitive blocks (M29, M30). M31 gives the tools to write such a graph as one Rust expression of existing blocks, without buffers or loops of its own: `Chain` for blocks in a row (also with a pair of streams between two stages), `Fork` for one input into two branches, automatic delay alignment of the branches, parameter paths and `ParamMap` for flat parameter names. `Chain` and `Fork` are themselves `Block`s, so they nest; they are heap-free and take their buffers from the same `Storage` as the blocks. `ParamMap` is a const table that holds no state. None of them is a pipeline kind. M32 rebuilds the existing composite blocks (`fm_stereo_demod` and others) from these tools.

### Chain with a pair between its stages

`Chain<A, B, S>` (see [Chains](#chains)) also accepts `A::Out = Two<X, Y>` with `X` and `Y` stream elements, and then `B::In = Two<X, Y>`. The bound is the sealed trait `ChainItem`, implemented for every `StreamElem` (one buffer) and for `Two<X, Y>` (two buffers of `chunk` samples each, written by `A` and consumed by `B` in lockstep, M26 §1). Everything else of the chain is unchanged: `process`, timing, parameters and `chain!`. `Frame` and port types that are not stream elements (`Packets`, M37) still do not compile as an intermediate. `Chain::storage_bytes(chunk)` counts both buffers of a pair, with the padding between them (see [Storage](#storage-of-compositions)).

### Fork

`Fork<A, B, S>` (module `fork.rs`) has one input and two branches: `A` and `B` read the same stream (`B::In = A::In`, a stream element), and the output is the pair `Two<A::Out, B::Out>` (both stream elements). A fork is usually followed in a `Chain` by a block that reads a pair, such as `make_complex` or `stereo_pll`. Forks nest only through such a join: a branch can be `Chain<Fork<..>, J>` with a joining block `J`, because the output of each branch must be a stream element and the fork's own output `Two<..>` is not one. A true three-way output (`Two<X, Two<Y, Z>>`) is not supported (D353).

- **Construction:** `Fork::new(a, b, chunk, storage)`. `chunk` is the capacity in samples of the fork's input buffer and of each branch's output buffer. `new` fails with `InvalidParameter` if `chunk` is 0 or smaller than the `min_input` of a branch (name `chunk`; unlike `Chain`, a fork does not raise `chunk`), if a branch has a ratio of kind `Variable`, if the two ratios differ, or if the delays differ by a fraction of an output sample (name `fork`, see [Delay alignment](#delay-alignment)); `Storage` if the storage cannot provide the buffers. `first()`, `second()`, `first_mut()` and `second_mut()` give access to the branches.
- **`process(input, (out_a, out_b))`:** the two outputs must have the same length (lockstep, M26 §1); otherwise the call is `LengthMismatch` and changes nothing. The fork copies input into its own input buffer and remembers how far each branch has read it. One pass of the loop (1) moves pairs to `(out_a, out_b)` while both branch buffers hold samples, (2) drops the input both branches have read and copies new input behind the rest, (3) calls each branch once on its unread part with the free part of its own output buffer (if it has room and either unread input or output it may still owe). The loop stops when the output is full or nothing moved. `Progress.consumed` is what the fork copied into its input buffer, `Progress.produced` the number of pairs written. The result does not depend on how the caller cuts input and output (R6.2), as far as the branches are invariant themselves.
- **Timing:** `ratio` that of either branch (they are equal); `kind` `Constant` (a `Variable` branch is rejected by `new`); `delay` the larger of the two delays, in input samples (after the alignment both outputs carry this delay); `min_input` the larger of the two; `map_index` that of the branch with the larger delay (the first branch on a tie); `reset` resets both branches, empties the buffers and fills the delay line with zeros.
- **Description:** `FORK_DESCRIPTION` has the kind `fork` and empty port and parameter lists, like `CHAIN_DESCRIPTION`. It is not a pipeline kind and not in `all_descriptions()`; `cargo xtask check` (names) lists `fork` as an exception next to `chain`.

### Delay alignment

With the common ratio `r` (output samples per input sample), the delays of the branches in output samples are `da = delay_a * r` and `db = delay_b * r`, computed exactly (`Delay` arithmetic). If `da == db`, nothing is inserted. If `da - db` is a positive integer `k`, the fork delays the output of branch `b` by `k` samples with an internal ring of `k` elements whose initial content is zero, so that both outputs are aligned; the other way round for `db > da`. At most one branch is delayed.

- A difference that is not a whole number of output samples is rejected with `InvalidParameter { name: "fork", reason: "fork branches are misaligned by a fraction of a sample" }`. Example: two `decimate` by 2 with 31 and 33 taps (delay 15 and 16 input samples, 7.5 and 8 output samples). In general, two branches with ratio 1/2 whose input delays differ by an odd number of samples are misaligned. A branch with a different ratio (for example a decimator against `delay` 0) is rejected earlier, with `"fork branches have different ratios"`. Such branches need an explicit `delay` stage in a branch that makes the difference whole, or a different design (for example, the decimation before the fork).
- IIR stages (`biquad`, `deemphasis`, `dc_block`) report a delay of 0 (M09), so they never cause an alignment: their group delay is not compensated. A branch with a FIR of 31 taps (delay 15) against `delay` 0 gets 15 samples of alignment on the `delay` branch.

### Parameter paths and `ParamMap`

Compositions name the parameters of inner blocks by path. A chain uses `0.` for its first and `1.` for its second stage, a fork `0.` for branch `a` and `1.` for branch `b`; nested chains and forks add their prefixes from the outside in (`"1.0.frequency_hz"` is parameter `frequency_hz` of branch `a` of a fork that is the second stage of a chain). A name without a valid prefix is `UnknownParameter`; the error of an inner block is returned unchanged.

A composite block that offers flat public names (like `audio_cutoff_hz` of `fm_stereo_demod`) keeps its own `set_param` and `param` and maps the flat name to a path with `ParamMap`: a `Copy` table of `(flat name, path)` entries, `&'static [(&'static str, &'static str)]`, built with `const fn ParamMap::new`. `path(flat)` returns the path (the first entry of a name wins), `set(&mut block, flat, value)` and `get(&block, flat)` forward the mapped path to the block's `set_param` and `param`, and a flat name that is not in the map is `UnknownParameter`. The composite's `description()` stays its own and lists the flat names, not the paths.

### Storage of compositions

All buffers come from the `Storage` given to `new`, after the buffers of the blocks that were built before. On an arena that starts at an aligned address (`AlignedBytes`) and is still empty, `new` takes exactly what `storage_bytes` returns; at a position that is not aligned for an element, up to `align_of - 1` bytes of padding per buffer come on top.

- `Chain::storage_bytes(chunk)`: `chunk * size_of::<A::Out>()` for a stream intermediate; for a pair `Two<X, Y>`, `chunk * size_of::<X>()`, the padding that aligns the second buffer behind the first, and `chunk * size_of::<Y>()`.
- `Fork::storage_bytes(&a, &b, chunk) -> Result<usize>`: in the order of allocation, an input buffer of `chunk` samples of `A::In`, an output buffer of `chunk` samples per branch (`A::Out`, then `B::Out`) and the alignment ring of `k` samples of the delayed branch's output type, with the padding between them. It returns the errors of `Fork::new` except `Storage`. Example: two `f32` branches, `chunk` 32, branch `b` delayed by 5: `(3 * 32 + 5) * 4` = 404 bytes.

See [embedded.md](embedded.md#sizing-the-arena-of-a-chain) for sizing a whole arena.

### Example: the stereo decoder

The stereo decoder of `fm_stereo_demod` (M24) as a composition: the multiplex signal goes straight on (`delay` 0) and through the 19 kHz band-pass, and the pair meets in the pilot loop. The listing is an illustration with the real constructor signatures; it is **not compiled** here (M32 builds it as a test). `qd`, `zero`, `pilot_bp`, `pll`, `audio_lp` and `deemph` are the configurations (`QuadDemodConfig`, `DelayConfig { samples: 0 }`, a `BiquadConfig` band-pass at 19 kHz, `StereoPllConfig`, a `DecimateConfig` and a `BiquadConfig` low-pass), `fs` the input rate and `fs_audio` the rate after the decimation; `st` is a `&mut` storage.

```rust
let decoder = chain!(st, 256;
    QuadDemodBlock::<C32, _>::new(&qd, fs, st)?,                 // C32 -> f32
    Fork::new(
        DelayRealBlock::<f32, _>::new(&zero, fs, st)?,           // multiplex, delay 0
        BiquadRealBlock::<f32, _>::new(&pilot_bp, fs, st)?,      // pilot band-pass, IIR: delay 0
        256,
        st,
    )?,                                                          // f32 -> Two<f32, f32>
    StereoPllBlock::<C32, _>::new(&pll, fs, st)?,                // Two<f32, f32> -> C32
    DecimateBlock::<C32, _>::new(&audio_lp, fs, st)?,            // C32 at fs -> fs_audio
    StereoMatrixBlock::<C32, _>::new(fs_audio, st)?,             // C32 -> C32 (left, right)
    BiquadBlock::<C32, _>::new(&deemph, fs_audio, st)?);         // de-emphasis of both channels
```

- **Types:** the chain between `quad_demod` and the fork passes `f32`; the chain between the fork and `stereo_pll` passes the pair `Two<f32, f32>` (two buffers of 256 `f32`).
- **Alignment:** both branches have the ratio 1 and the delay 0 (`delay` 0 and an IIR band-pass), so the fork inserts nothing.
- **Parameter paths:** `chain!` nests to the right, so the stages are `0` (`quad_demod`), `1.0` (the fork, with branches `1.0.0` and `1.0.1`), `1.1.0` (`stereo_pll`), `1.1.1.0` (`decimate`), `1.1.1.1.0` (`stereo_matrix`) and `1.1.1.1.1` (de-emphasis). A composite block over this expression can offer flat names with `ParamMap::new(&[("stereo", "1.1.0.stereo"), ("pilot_q", "1.0.1.q")])`.
- **Storage of the tools:** the fork takes three buffers of 256 `f32` (3072 bytes, no ring); the five joints of the chain take 256 `f32`, 2 x 256 `f32`, and three times 256 `C32` (1024 + 2048 + 3 x 2048 bytes).

## Filter design

`sdrk_blocks::design` computes in `f64` and writes into caller slices; the functions mirror SciPy so that golden vectors can check them (`tests/golden/sdrk_golden/design.py` calls SciPy and is the reference):

- `firwin_lowpass(cutoff_hz, fs_hz, window, taps)` is `scipy.signal.firwin(N, cutoff, window=…, pass_zero=True, scale=True, fs=fs)` with a symmetric Hann, Hamming, Blackman or Kaiser window and a DC gain of 1 (`0 < cutoff < fs / 2`, else `OutOfRange`; a window that leaves no DC gain, such as a two-point Hann window, is `InvalidParameter`).
- `kaiserord(ripple_db, transition_hz, fs_hz)` is `scipy.signal.kaiserord(ripple, 2 * transition / fs)` with the tap count forced odd; `kaiser_beta(a)` is `scipy.signal.kaiser_beta`.
- `halfband_len(transition_hz, attenuation_db, fs_hz)` is `kaiserord` rounded up to the next `N` with `N mod 4 == 3`; `halfband_taps(beta, taps)` is `firwin_lowpass(fs / 4, fs, Kaiser { beta })` for `fs = 1` with every second tap around the centre (`h[m +- 2 i]`) set to exactly `0.0`, the right half mirrored from the left one (exactly symmetric, D72) and the sum scaled to 1; about half of the taps are zero and the outer taps are not. The golden cases `halfband_n{11,23,63}` and `halfband_len_table` compare with SciPy (1e-12, exact counts).
- `to_f32` and `to_q15` convert the taps; `to_q15` refuses a tap above `1 - 2^-15` by more than 1 % (it would saturate badly).
- `DesignSpec { cutoff_hz, transition_hz, attenuation_db }` is the parameter set of the filtering blocks. A block designs `kaiserord` + `firwin_lowpass(Kaiser)` one tap at a time (no temporary `f64` buffer), at most `max_taps` taps (default 1023, `OutOfRange` beyond).

The golden tests compare the taps with SciPy to 1e-12 and the tap counts exactly.

The IIR designs for the block `biquad` (all in `f64`, section coefficients `[b0, b1, b2, a1, a2]` with `a0 = 1`, i.e. one row of a SciPy `sos` matrix without the 1):

- `biquad_coefs(response, f0_hz, q, fs_hz)` is the RBJ Audio EQ Cookbook (R. Bristow-Johnson) section for `BiquadResponse::{Lowpass, Highpass, Bandpass, Notch}`, normalised by `a0`. With `0 < f0 < fs / 2` and `q > 0` (else `OutOfRange`), `w0 = 2 pi f0 / fs`, `c = cos w0` and `alpha = sin(w0) / (2 q)`: low-pass `b = [(1 - c) / 2, 1 - c, (1 - c) / 2]`, high-pass `b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]`, band-pass (0 dB peak gain) `b = [alpha, 0, -alpha]`, notch `b = [1, -2 c, 1]`, all with `a = [1 + alpha, -2 c, 1 - alpha]`.
- `butterworth_sos(response, order, cutoff_hz, fs_hz, sos)` is a Butterworth low-pass or high-pass of even `order` 2 to 8 as `order / 2` sections: section `k = 1 ... order / 2` is `biquad_coefs(response, cutoff_hz, Q_k, fs_hz)` with `Q_k = 1 / (2 cos((2 k - 1) pi / (2 order)))`, lowest Q first. Both the cookbook and `scipy.signal.butter` use the bilinear transform prewarped at the cutoff, so the cascade has the response of `scipy.signal.butter(order, cutoff_hz, btype, fs=fs_hz, output='sos')`; the generator asserts this within 1e-9 at 512 frequencies and the gain at the cutoff is -3.0103 dB.
- `to_q14` rounds to Q2.14 with `q14_from_f64` (`floor(x * 16384 + 0.5)`; a coefficient outside -2 ... 2 - 2^-14 is an error), `to_f32_sos` converts to `f32`.
- `preemphasis_coefs(tau_s, fs_hz)` is the first-order pre-emphasis `[b0, b1, a1]` of `H(z) = (b0 + b1 z^-1) / (1 + a1 z^-1)` with unity gain at DC: `H(s) = (1 + s tau) / (1 + s tau_p)` with a pole at `f_p = fs / 2.4` (`tau_p = 1 / (2 pi f_p)`, 20 kHz at 48 kHz) by the bilinear transform without prewarping; with `K = 2 fs` and `den = 1 + K tau_p`: `b0 = (1 + K tau) / den`, `b1 = (1 - K tau) / den`, `a1 = (1 - K tau_p) / den`. `tau_s` must lie in `(0, 1e-3]` (`PREEMPHASIS_MAX_TAU_S`) and `fs_hz` must be at least 40 000 (`PREEMPHASIS_MIN_RATE_HZ`), else `OutOfRange`. The magnitude differs from `|1 + j 2 pi f tau|` by at most 0.2 dB from 30 Hz to `min(15 kHz, 0.45 fs)` (measured at most 0.17 dB for 44.1, 48, 96 and 240 kHz with 50, 75 and 530 us, D184); at 32 kHz the error reaches 1 dB, hence the lower limit. The gain at Nyquist is `tau / tau_p` (16 dB for 50 us at 48 kHz, 36 dB for 530 us at 48 kHz).

## Descriptions

Every block has a `BlockDescription` (kind, summary, details, ports, parameters, required parameters) as `static` data; `all_descriptions()` lists all thirty-six. A `ParamDesc` has a name that ends in its unit (`_hz`, `_db`, `_us`), a `Unit`, a `ParamKind` with absolute limits and a default, a `typical` text, `Mutability::Build` or `Runtime`, and a description. Rules, all enforced by `crates/blocks/sdrk-blocks/tests/describe.rs`:

- every text is non-empty and has no leading or trailing whitespace; a summary is one sentence of at most 120 characters that ends with a period;
- block kinds and parameter names are `snake_case` and unique; a unit suffix in the name matches the `Unit`;
- defaults lie inside the range; a parameter without a default is either listed in `required` or documents a rate-dependent default (its `typical` or `description` text says "default");
- the config defaults equal the description defaults, `set` and `get` accept and return every described parameter, and every runtime parameter is accepted by `set_param` and readable by `param`;
- `min`/`max` are the absolute limits a JSON schema can check; limits that depend on the sample rate are applied by `validate` and `new`.

The tables below are generated: `cargo test -p sdrk-blocks --test describe -- --ignored print --nocapture` prints them and the test `docs_blocks_md_contains_the_generated_table` fails when this file differs.

<!-- blocks-table:begin -->

| Kind | Summary | Inputs | Outputs | Required |
| --- | --- | --- | --- | --- |
| `convert` | Converts raw device bytes in a sample format to complex or real samples. | `in`: raw | `out`: c32, cq15, f32, q15 | `format` |
| `convert_out` | Converts complex samples to raw device bytes in a sample format. | `in`: c32, cq15 | `out`: raw | `format` |
| `scale` | Multiplies a complex signal by a gain given in decibels. | `in`: c32, cq15 | `out`: c32, cq15 | - |
| `dc_block` | Removes the DC offset of a complex signal with a one-pole high-pass. | `in`: c32, cq15 | `out`: c32, cq15 | - |
| `fir` | Low-pass FIR filter designed from cutoff, transition width and attenuation. | `in`: c32, cq15 | `out`: c32, cq15 | `cutoff_hz`, `transition_hz` |
| `ddc` | Shifts a signal to 0 Hz, low-pass filters it and decimates it. | `in`: c32, cq15 | `out`: c32, cq15 | `decimation` |
| `duc` | Interpolates a signal, removes the images and shifts it up to a carrier frequency. | `in`: c32, cq15 | `out`: c32, cq15 | `interpolation` |
| `fm_demod` | Demodulates an FM signal to audio with de-emphasis and decimation. | `in`: c32, cq15 | `audio`: f32, q15 | - |
| `preemph_limit` | Pre-emphasis and peak limiter for the audio of an FM transmitter. | `in`: f32, q15 | `out`: f32, q15 | - |
| `fm_mod` | Frequency-modulates a real signal onto a complex baseband carrier. | `in`: f32, q15 | `out`: c32, cq15 | - |
| `fm_stereo_demod` | Demodulates a broadcast FM signal to stereo audio (left in re, right in im). | `in`: c32, cq15 | `audio`: c32, cq15 | - |
| `resample` | Changes the sample rate by a rational factor, optionally tunable in ppm. | `in`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | `rate_out_hz` |
| `spectrum` | Computes averaged power spectra in dB relative to full scale as frames. | `in`: c32, cq15 | `frame`: frame(fft_size) | - |
| `squelch` | Passes a complex signal only while its average power is above a threshold. | `in`: c32, cq15 | `out`: c32, cq15 | - |
| `agc` | Automatic gain control: scales a complex signal to a constant output amplitude. | `in`: c32, cq15 | `out`: c32, cq15 | - |
| `am_demod` | Demodulates an AM signal to audio: envelope, DC removal and decimation. | `in`: c32, cq15 | `audio`: f32, q15 | - |
| `ssb_demod` | Demodulates a single-sideband signal (USB or LSB) to audio. | `in`: c32, cq15 | `audio`: f32, q15 | - |
| `biquad` | Filters with a Butterworth low-pass or high-pass, a band-pass or a notch (IIR). | `in`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | `cutoff_hz` |
| `halfband_decimate` | Halfband low-pass filter and decimation by 2. | `in`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | - |
| `real_to_iq` | Turns a real signal into a complex one at half the rate (fs/4 shift and halfband). | `in`: f32, q15 | `out`: c32, cq15 | - |
| `quad_demod` | Quadrature (FM) discriminator: the phase step between consecutive samples. | `in`: c32, cq15 | `out`: f32, q15 | - |
| `deemphasis` | Single-pole low-pass that undoes the pre-emphasis of FM broadcast audio. | `in`: f32, q15 | `out`: f32, q15 | - |
| `rotate` | Shifts the spectrum of a complex signal by a frequency. | `in`: c32, cq15 | `out`: c32, cq15 | `frequency_hz` |
| `decimate` | Low-pass filter and decimation by an integer factor. | `in`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | `factor` |
| `interpolate` | Interpolation by an integer factor with an image-rejection low-pass. | `in`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | `factor` |
| `mag` | Magnitude (envelope) of complex samples. | `in`: c32, cq15 | `out`: f32, q15 | - |
| `delay` | Delays a signal by a whole number of samples. | `in`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | `samples` |
| `add` | Adds two streams of the same type, sample by sample. | `a`: c32, cq15, f32, q15; `b`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | - |
| `mul` | Multiplies two streams of the same type, sample by sample. | `a`: c32, cq15, f32, q15; `b`: c32, cq15, f32, q15 | `out`: c32, cq15, f32, q15 | - |
| `make_complex` | Combines a real and an imaginary stream into one complex stream. | `re`: f32, q15; `im`: f32, q15 | `out`: c32, cq15 | - |
| `to_complex` | Turns a real stream into a complex stream with a zero imaginary part. | `in`: f32, q15 | `out`: c32, cq15 | - |
| `real_part` | Outputs the real part of a complex stream as a real stream. | `in`: c32, cq15 | `out`: f32, q15 | - |
| `imag_part` | Outputs the imaginary part of a complex stream as a real stream. | `in`: c32, cq15 | `out`: f32, q15 | - |
| `nco` | Generates a complex tone at frequency_hz, one sample per input sample. | `clock`: c32, cq15, f32, q15 | `out`: c32, cq15 | `frequency_hz` |
| `stereo_pll` | Locks a loop to the 19 kHz stereo pilot and demodulates the 38 kHz difference. | `mpx`: f32, q15; `pilot`: f32, q15 | `out`: c32, cq15 | - |
| `stereo_matrix` | Turns sum and difference (S, D) into left and right: L = S + 2 D, R = S - 2 D. | `in`: c32, cq15 | `out`: c32, cq15 | - |

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
| `preemph_limit` | `preemphasis_us` | us | build | f64, 0 to 1000, default 50 | 50 (Europe), 75 (North America), 530 for two-way radio, 0 for none |
| `preemph_limit` | `limit` | - | runtime | f64, 0.1 to 1, default 1 | 1.0 before fm_mod, 0.85 when a filter follows |
| `preemph_limit` | `hold_s` | s | build | f64, 0 to 1, default 0.02 | 0.02 (covers 50 Hz) |
| `preemph_limit` | `release_s` | s | build | f64, 0.001 to 10, default 0.2 | 0.2 |
| `fm_mod` | `deviation_hz` | Hz | runtime | f64, 0 to 1000000000000, default 75000 | 75e3 for FM broadcast, 5e3 for narrow-band voice |
| `fm_stereo_demod` | `deviation_hz` | Hz | build | f64, 1 to 1000000000000, default 75000 | 75e3 for broadcast FM |
| `fm_stereo_demod` | `deemphasis_us` | us | build | f64, 0 to 1000000, default 50 | 50 in Europe, 75 in North America, 0 to switch off |
| `fm_stereo_demod` | `audio_cutoff_hz` | Hz | build | f64, 1 to 16000, default 15000 | 15e3 for broadcast FM |
| `fm_stereo_demod` | `audio_decimation` | - | build | integer, 1 to 64, default 10 | 10 to go from 480 kHz to 48 kHz |
| `fm_stereo_demod` | `stereo` | - | runtime | choice auto / mono, default auto | auto; mono for a weak station |
| `fm_stereo_demod` | `pll_natural_hz` | Hz | build | f64, 1 to 190, default 20 | 20 |
| `resample` | `rate_out_hz` | Hz | build | f64, 1 to 1000000000000, no default | 48000 for audio |
| `resample` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.45 times the lower of the input and output rates |
| `resample` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the lower of the input and output rates |
| `resample` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `resample` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023; 8192 for 48000 to 44100 in the rational mode |
| `resample` | `tunable` | - | build | flag, default false | false; true to follow a drifting clock |
| `resample` | `ppm` | ppm | runtime | f64, -1000 to 1000, default 0 | 0; +100 makes the output rate 100 ppm higher |
| `spectrum` | `fft_size` | - | build | integer, 16 to 65536, default 1024 | 1024; 4096 for finer resolution |
| `spectrum` | `window` | - | build | choice hann / blackman_harris / kaiser, default hann | hann |
| `spectrum` | `kaiser_beta` | - | build | f64, 0 to 50, default 8.6 | 8.6 |
| `spectrum` | `averaging` | - | runtime | integer, 1 to 1024, default 1 | 1; 10 for a smoother display |
| `spectrum` | `interval_samples` | samples | runtime | integer, 1 to 1073741824, no default | default fft_size; fft_size / 2 for 50 % overlap |
| `squelch` | `threshold_db` | dB | runtime | f64, -150 to 0, default -40 | -40 for a strong local signal, -55 for weak signals |
| `squelch` | `hysteresis_db` | dB | runtime | f64, 0 to 40, default 3 | 3 |
| `squelch` | `tau_s` | s | build | f64, 0.000001 to 100, default 0.01 | 0.01 at 48 kHz |
| `agc` | `reference_db` | dB | runtime | f64, -30 to 0, default -6 | -6 (output amplitude 0.5) |
| `agc` | `attack_s` | s | runtime | f64, 0.000001 to 100, default 0.005 | 0.005 for SSB, 0.1 for AM |
| `agc` | `release_s` | s | runtime | f64, 0.000001 to 100, default 0.2 | 0.2 for SSB, 0.5 for AM |
| `agc` | `min_gain_db` | dB | runtime | f64, -40 to 90, default -20 | -20 |
| `agc` | `max_gain_db` | dB | runtime | f64, -40 to 90, default 60 | 60 |
| `agc` | `initial_gain_db` | dB | build | f64, -40 to 90, default 0 | 0 |
| `am_demod` | `dc_cutoff_hz` | Hz | build | f64, 0 to 1000000000000, default 30 | 30 for broadcast, 0 to switch off |
| `am_demod` | `audio_cutoff_hz` | Hz | build | f64, 1 to 1000000000000, default 5000 | 5e3 for broadcast, 2.5e3 for airband voice |
| `am_demod` | `audio_decimation` | - | build | integer, 1 to 64, default 1 | 1 at 48 kHz, 6 with audio_cutoff_hz 2500 to go to 8 kHz |
| `ssb_demod` | `sideband` | - | runtime | choice usb / lsb, default usb | usb; lsb for amateur radio below 10 MHz |
| `ssb_demod` | `low_hz` | Hz | build | f64, 0 to 1000000000000, default 300 | 300 for voice |
| `ssb_demod` | `high_hz` | Hz | build | f64, 0 to 1000000000000, default 3000 | 3000 for voice |
| `ssb_demod` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, default 200 | 200 |
| `ssb_demod` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `ssb_demod` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023 |
| `ssb_demod` | `audio_decimation` | - | build | integer, 1 to 64, default 1 | 1 at 48 kHz, 6 to go to 8 kHz |
| `biquad` | `response` | - | build | choice lowpass / highpass / bandpass / notch, default lowpass | lowpass for hiss, highpass to remove a sub-tone, notch for a whistle |
| `biquad` | `cutoff_hz` | Hz | runtime | f64, 0.000001 to 1000000000000, no default | 4000 for an AM audio low-pass, 300 to remove CTCSS |
| `biquad` | `order` | - | build | integer, 2 to 8, default 2 | 4 |
| `biquad` | `q` | - | runtime | f64, 0.001 to 1000, default 0.7071 | 2 for a band-pass, 10 for a narrow notch |
| `halfband_decimate` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the input rate |
| `halfband_decimate` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `halfband_decimate` | `max_taps` | - | build | integer, 3 to 65536, default 1023 | 1023 |
| `real_to_iq` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the input rate |
| `real_to_iq` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `real_to_iq` | `max_taps` | - | build | integer, 3 to 65536, default 1023 | 1023 |
| `quad_demod` | `deviation_hz` | Hz | build | f64, 0.000001 to 1000000000000, default 75000 | 75e3 for broadcast FM, 5e3 for narrow-band voice |
| `deemphasis` | `tau_us` | us | build | f64, 1 to 10000, default 50 | 50 in Europe, 75 in North America |
| `rotate` | `frequency_hz` | Hz | runtime | f64, -1000000000000 to 1000000000000, no default | -600e3 to bring a carrier 600 kHz above the tuner centre to 0 Hz |
| `decimate` | `factor` | - | build | integer, 1 to 1024, no default | 10 to go from 2.4 MHz to 240 kHz |
| `decimate` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | 100e3 for FM broadcast; default 0.4 times the output rate |
| `decimate` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the output rate |
| `decimate` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `decimate` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023 |
| `interpolate` | `factor` | - | build | integer, 1 to 1024, no default | 10 to go from 240 kHz to 2.4 MHz |
| `interpolate` | `cutoff_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.4 times the input rate |
| `interpolate` | `transition_hz` | Hz | build | f64, 0.000001 to 1000000000000, no default | default 0.1 times the input rate |
| `interpolate` | `attenuation_db` | dB | build | f64, 8 to 200, default 60 | 60 |
| `interpolate` | `max_taps` | - | build | integer, 1 to 65536, default 1023 | 1023 |
| `delay` | `samples` | samples | build | integer, 0 to 65536, no default | 63 to match the group delay of a 127-tap linear-phase filter |
| `nco` | `frequency_hz` | Hz | runtime | f64, -1000000000000 to 1000000000000, no default | 19e3 for the stereo pilot at 192 kHz |
| `nco` | `harmonic` | - | build | integer, 1 to 8, default 1 | 1; 2 for the 38 kHz subcarrier of a 19 kHz pilot |
| `nco` | `phase_turns` | - | build | f64, 0 to 1, default 0 | 0; 0.25 starts at +j |
| `stereo_pll` | `pilot_hz` | Hz | build | f64, 1 to 1000000000000, default 19000 | 19e3 for broadcast FM |
| `stereo_pll` | `natural_hz` | Hz | build | f64, 1 to 190, default 20 | 20 |
| `stereo_pll` | `pilot_level` | - | build | f64, 0.000001 to 1, default 0.1 | 0.1; for Q15 0.1 times the scale of the discriminator output |
| `stereo_pll` | `stereo` | - | runtime | choice auto / mono, default auto | auto; mono for a weak station |

<!-- blocks-table:end -->

## Names

A *kind* is the name that a flow file uses for a block (`"kind": "fm_demod"`) and that the description, the schema and the WebAssembly bindings show to users. Every kind follows this scheme (M24.1); `cargo xtask check` checks rules 1 and 9 (check `names`).

1. **Form:** lowercase ASCII words joined by `_`, matching `^[a-z][a-z0-9]*(_[a-z0-9]+)*$`, at most 32 characters (the node-name pattern of the pipeline is wider; kinds are stricter).
2. **Operations are verbs or established operation nouns** and say what happens to the signal: `convert`, `scale`, `resample`, `halfband_decimate`, `decimate`, `interpolate`, `rotate`, `delay`, `add`, `mul`.
3. **Modulation pairs are `<method>_demod` and `<method>_mod`:** `fm_demod`/`fm_mod`, `am_demod`/`am_mod`, `ssb_demod`/`ssb_mod`, `fm_stereo_demod`/`fm_stereo_mod`. A receiver-only or transmitter-only block still carries its direction, so that the pair can be added later without a rename.
4. **Line coding and framing are `<protocol>_decode` and `<protocol>_encode`:** blocks that turn symbols or bits into packets and back (`rds_decode`/`rds_encode`). `_demod`/`_mod` work on samples on both sides, `_decode`/`_encode` have bits or packets on one side.
5. **Filters name their construction** and take the response as a parameter: `fir`, `biquad`. A filter with one fixed purpose names the purpose: `dc_block`, `deemphasis`.
6. **Established abbreviations stay:** `ddc`, `duc`, `agc`, `nco`, `pll` (inside a name such as `stereo_pll`) and `iq` (in-phase and quadrature, the complex baseband signal; `real_to_iq`). The `summary` of the description writes them out once ("Digital down converter (ddc): ...").
7. **No two kinds that differ only by a prefix or suffix when they do different things.** A name that hides the rate change, the direction or the domain change of a block is wrong.
8. **Kernels keep their own names.** Kernel names in `sdrk-core` name the mathematical operation and are part of the golden vectors, benches and limits; they are not renamed with the blocks. A block that wraps exactly one kernel and adds nothing a user sees takes the kernel's name when that name fits the rules above (`halfband_decimate`, `biquad`, and from M25 `preemph_limit`); otherwise it takes its own name (`real_to_iq` wraps the kernel `real_to_complex`).
9. **Rust names follow the kind:** the module and file are `blocks/<kind>.rs`; the block type is `<Kind>Block` with `<Kind>` the kind in UpperCamelCase (`fm_stereo_demod` -> `FmStereoDemodBlock`, `real_to_iq` -> `RealToIqBlock`, `dc_block` -> `DcBlockBlock`); the variant for real samples is `<Kind>RealBlock`, and a kind that has only a real variant defines only `<Kind>RealBlock` (`deemphasis` -> `DeemphasisRealBlock`); a configuration struct is `<Kind>Config`; constants of the block start with the kind in upper case (`FM_STEREO_DEMOD_CHUNK`); the pipeline kind type in `sdrk-pipeline` is `<Kind>Kind`. Abbreviations are one word in UpperCamelCase (`Iq`, `Ddc`, `Nco`).
10. **Building blocks of a composite block** (for example `stereo_pll`) follow the same rules; they are ordinary kinds.

M24.1 renamed four kinds. The old names are gone: the pipeline reports them as `unknown_kind`, and the Rust crates have only the new names (no aliases before release 0.1).

| Old kind | New kind | Old Rust names | New Rust names |
| --- | --- | --- | --- |
| `resampler` | `resample` | `ResamplerBlock`, `ResamplerRealBlock`, `ResamplerConfig`, `ResamplerKind` | `ResampleBlock`, `ResampleRealBlock`, `ResampleConfig`, `ResampleKind` |
| `halfband` | `halfband_decimate` | `HalfbandBlock`, `HalfbandRealBlock`, `HalfbandConfig`, `HalfbandKind` | `HalfbandDecimateBlock`, `HalfbandDecimateRealBlock`, `HalfbandDecimateConfig`, `HalfbandDecimateKind` |
| `real_to_complex` | `real_to_iq` | `RealToComplexBlock`, `RealToComplexKind` | `RealToIqBlock`, `RealToIqKind` |
| `fm_stereo` | `fm_stereo_demod` | `FmStereoBlock`, `FmStereoConfig`, `FM_STEREO_CHUNK`, `FmStereoKind` | `FmStereoDemodBlock`, `FmStereoDemodConfig`, `FM_STEREO_DEMOD_CHUNK`, `FmStereoDemodKind` |

The kinds that later milestones add have their names reserved here, so that every milestone uses names that fit the scheme. A milestone that needs a name not in the table chooses it by the rules above and adds it to the table.

| Kind | Purpose | Planned in |
| --- | --- | --- |
| `stereo_pll`, `stereo_matrix` | building blocks of `fm_stereo_demod` | M30 (delivered) |
| `to_complex`, `real_part`, `imag_part`, `make_complex` | domain changes: real → complex with imaginary part 0, complex → real part, complex → imaginary part, two real streams → one complex stream | M30 (delivered) |
| `nco`, `add`, `mul` | oscillator source, sum and product of two streams | M30 (delivered; pipeline kinds of `to_complex`, `real_part`, `imag_part`, `nco`, `stereo_matrix` in M30, of `add`, `mul`, `make_complex`, `stereo_pll`: M34) |
| `noise_blank`, `auto_notch` | impulse blanker, adaptive notch | M33 |
| `am_mod`, `ssb_mod` | AM and SSB transmit | M35 |
| `fm_stereo_mod` | FM stereo encoder (pilot, 38 kHz subcarrier) | M36 |
| `frame_encode`, `frame_decode` | test framing of `sdrk-proto` (sync word, length, CRC) | M37 |
| `symbol_sync`, `slice` | symbol timing recovery; symbol to soft bits | M38 |
| `fsk_demod`/`fsk_mod`, `afc` | FSK, GFSK, GMSK, AFSK (parameters, not kinds); automatic frequency control | M39 |
| `ook_demod`/`ook_mod`, `hdlc_decode`/`hdlc_encode`, `adsb_decode` | on-off keying, HDLC framing, Mode S/ADS-B | M39.1 |
| `costas`, `psk_demod`/`psk_mod` | carrier recovery (parameter `order` 2/4/8); PSK with variants (QPSK, DQPSK, 8PSK, π/4-DQPSK, OQPSK) as parameters | M40, M40.1 |
| `rds_demod`, `rds_decode` | RDS subcarrier to bits, bits to groups | M41 |
| `rds_encode`, `fm_stereo_rds_mod` | RDS groups to the shaped baseband; FM stereo encoder with RDS | M42 |
| `equalize` | adaptive equalizer | M43 |
| `qam_demod`/`qam_mod` | 16-QAM and 64-QAM | M44 |
| `sync_detect` | sample-level sync-word correlator | backlog |

## The blocks

- **`convert` / `convert_out`:** raw device bytes to `C32`/`CQ15` and back (all interleaved complex formats of [device.md](device.md)). The block works on whole packing groups only (for example 5 bytes for two 10-bit complex samples); `Progress` counts bytes on the raw side and samples on the other. `convert` also accepts the formats with `Layout::Real` (`rs16`, `rf32`, ...): `ConvertRealBlock<T>` (`f32`, `Q15`) calls `convert_real` and has the same parameter `format`; `ConvertBlock` refuses a real format and `ConvertRealBlock` a complex one, the pipeline kind and the automatic conversion of a raw input choose the block by the layout of the format (the port type of the node is then `f32`/`q15`). For a real format a packing group counts real samples (`rs16`: 2 bytes per sample). `convert_out` has no real variant. Planar formats cannot be streamed.
- **`scale`:** `gain_db` (runtime) as a real gain, `-120` to `+40` dB for `C32` and `-120` to `0` dB for `CQ15` (0 dB is the largest Q15 value, about -0.0003 dB).
- **`dc_block`:** `r = exp(-2 pi cutoff / fs)` (rounded to Q15 for `CQ15`).
- **`fir`:** a Kaiser-window low-pass from `cutoff_hz`, `transition_hz`, `attenuation_db` (60) and `max_taps` (1023); `cutoff_hz` and `transition_hz` are required.
- **`ddc`:** an NCO at `-frequency_hz` shifts a signal at `+frequency_hz` to 0 Hz, the designed low-pass (default cutoff `0.4 * fs / decimation`, transition `0.1 * fs / decimation`) and the fused `ddc` kernel with a scratch buffer of `max_chunk` samples (default 4096). `frequency_hz` is a runtime parameter and keeps the oscillator phase continuous.
- **`duc`:** the fused `duc` kernel with taps designed at the *output* rate and multiplied by `L`, the NCO at `+frequency_hz` at the output rate; defaults relative to the input rate (cutoff `0.4`, transition `0.1`). `max_chunk` is counted in output samples and must be at least `interpolation`.
- **`fm_demod`:** `quad_demod` + `deemphasis` (the fused `fm_demod` kernel; `deemphasis_us = 0` switches it off) + the audio low-pass with decimation (`resample_real`, L = 1). The discriminator gain is `fs / (2 pi deviation_hz)` for `C32` (the full deviation reads 1.0) and the largest Q15 value for `CQ15` (output unit phase per sample divided by pi; 75 kHz at 240 kHz reads about 0.625). The audio filter has the cutoff `audio_cutoff_hz`, a transition width of a quarter of it and 60 dB; its stop band must lie below the output Nyquist frequency. The block works through its input in pieces of `FM_DEMOD_CHUNK` (256) samples.
- **`preemph_limit`:** `PreemphLimitBlock<T>` (`f32`, `Q15`) around the kernel `preemph_limit` (bit-exact in both types, not accelerated): the pre-emphasis of `preemphasis_coefs(preemphasis_us, fs)` (`preemphasis_us = 0` gives `[1, 0, 0]` and works at any rate; a pre-emphasis at a rate below 40 kHz is `OutOfRange`) and a peak limiter with instant attack, `hold_s` of constant gain and a release with the time constant `release_s`, in one pass; no output sample exceeds `limit` times full scale (in Q15 the rounded raw limit; the value -32768 never leaves the block, D179). `limit` is a runtime parameter: `set_param("limit")` calls `set_limit`, which applies from the next sample and keeps the envelope and the hold. `rate_out` equals `rate_in`, the delay is 0, the block needs no storage, and `gain()` reports the current gain reduction (1.0 while idle). **Position:** directly before `fm_mod`, at the rate of the modulator. Any filter or resampler between the limiter and the modulator can raise the peaks again: measured while planning, limiting at 48 kHz and then resampling to 240 kHz (the `fm_transmitter` chain) let overdriven noise reach 1.41 times the limit, while limiting at 240 kHz kept it at exactly 1.0 (the loop test TX-PEAK measures 0.998 with the `shape` node of `fm_transmitter`). When a channel filter must follow the limiter (the voice band of `nfm_transmitter`), set `limit` below 1 to leave room for its overshoot (0.85 there). A finite `f32` input of about `1e36` or more is outside the documented range (D178).
- **`fm_mod`:** `fm_mod` with a runtime `deviation_hz`; the phase stays continuous when it changes.
- **`squelch`:** a power gate (`squelch` kernel, bit-exact in both types). The average power `p` follows `p += alpha (|x|^2 - p)` with `alpha = 1 - exp(-1 / (fs tau_s))`; the gate opens when `p >= 10^(threshold_db / 10)` and closes when `p < 10^((threshold_db - hysteresis_db) / 10)`, otherwise it keeps its state. Closed output is exactly zero, so a following `fm_demod` outputs exact silence. `threshold_db` and `hysteresis_db` are runtime parameters (`set_param` calls `set_thresholds`; the average power and the gate state are kept and the new levels apply from the next sample), `tau_s` is a build parameter. **Chain position:** the squelch works on the channel signal, so it belongs **after** the channel filter (`ddc`) and **before** the demodulator and any gain control; after an AGC every signal, noise included, has the same level and the gate would always open. Example: `ddc` -> `squelch` -> `fm_demod` in `examples/flows/nfm_receiver.json`. **Channel filter for narrow-band FM:** set the `ddc` cutoff to the deviation plus the highest audio frequency (about 3 kHz), half the necessary bandwidth of the emission: `cutoff_hz` 8000 (`transition_hz` 4000) for 25 kHz channels with 5 kHz deviation, 6000 (3000) for 12.5 kHz channels with 2.5 kHz deviation. A narrower filter cuts the FM sidebands, and the distortion (mostly the third harmonic of the audio) limits the SINAD: 6000 Hz with 5 kHz deviation gives about 22 dB instead of 35 dB.
- **`agc`:** automatic gain control (`agc` kernel; `C32` tolerance class, `CQ15` bit-exact). The gain `g` multiplies every sample, then, with `m = |y|`, `ref = 10^(reference_db / 20)` and `r` the attack rate (`1 - exp(-1 / (fs attack_s))`) when the output is too loud and the release rate (`release_s`) when it is too quiet, the gain is corrected by about `g <- g (1 + r (ref - m) / m)` (the error normalised by the output level, as the `agc3_cc` of GNU Radio; the divisor is limited to `ref / 4` ... `1.5 ref`, the exact update is in [kernels.md](kernels.md)); `g` stays within `min_gain_db` and `max_gain_db`. The loop is first order with the time constant `attack_s` or `release_s` at any level (a step of 30 dB settles within 1 dB after 5 time constants), and a sample of exactly zero (a closed squelch) holds the gain. `reference_db`, `attack_s`, `release_s`, `min_gain_db` and `max_gain_db` are runtime parameters (`set_param` calls `set_params`; the current gain is kept and clamped to the new range, a rejected value changes nothing), `initial_gain_db` is a build parameter and the gain `reset` returns to. `CQ15` keeps the gain as a Q.32 integer and the loop in integer arithmetic. **AGC time constants for AM:** the loop must be slow against the audio period, otherwise it follows the modulation and distorts it: attack 0.1 s and release 0.5 s for AM (`examples/flows/am_receiver.json`). The defaults (0.005 s, 0.2 s) suit SSB (M21). Chain position: after the channel filter and a squelch, before the demodulator.
- **`am_demod`:** the envelope detector `mag`, the DC blocker `dc_block_real` with the pole `r = exp(-2 pi dc_cutoff_hz / fs)` (`dc_cutoff_hz` 0 switches it off) and the audio low-pass with decimation (`resample_real`, L = 1, M = `audio_decimation`), the filter designed exactly as in `fm_demod` (transition width a quarter of `audio_cutoff_hz`, 60 dB; with `audio_decimation` 1 there is no filter). The audio is the envelope without its mean: a carrier of amplitude `A` with modulation index `m` reads `m A`, so the AGC in front sets the audio level (carrier at the AGC reference `ref` gives `m ref`). Works in blocks of 256 samples through two work buffers in the storage. **Chain order for AM:** `ddc -> squelch -> agc -> am_demod`. **Silence:** an input sample of exactly zero (the output of a closed squelch, digital silence) gives an envelope of exactly zero, which the block writes as zero audio and which clears the DC blocker, so a gated channel is silent at once (no decaying tail; with `audio_decimation` above 1 the audio filter rings out for its length). The runs of non-zero envelope between such samples go through the `dc_block_real` kernel. The `agc` in front holds its gain on zero input samples, so the first loud samples of the next transmission meet the gain of the last one and are corrected within a few samples (the attack step grows with the error).
- **`ssb_demod`:** a single-sideband demodulator for `C32` and `CQ15` that needs no kernel of its own. With `c = (low_hz + high_hz) / 2`, `b = (high_hz - low_hz) / 2` and `s = +1` for USB, `-1` for LSB: (1) `rotate` shifts the input by `-s c` (NCO at the input rate), (2) `fir_decimate` low-passes with the cutoff `b` (taps from `kaiserord` and a Kaiser `firwin` at the input rate with `transition_hz` and `attenuation_db`, quantized as in the `fir` block) and decimates by `audio_decimation`, (3) a second `rotate` shifts back by `+s c` (NCO at the output rate) and (4) the output is the real part of every sample. The sideband `[low_hz, high_hz]` above (USB) or below (LSB) the carrier at 0 Hz comes out as audio from `low_hz` to `high_hz`: a tone at the carrier `+ f` (USB) gives an audio tone at `f` with the input amplitude, and the opposite sideband is removed by the filter (60 dB at the defaults). `sideband` (`usb`/`lsb`) is a runtime parameter: `set_param` flips the sign of both NCO increments from the next call, both phases continue, so the audio stays continuous; both NCOs start at phase 0 after construction and `reset`. Validation: `0 <= low_hz < high_hz` and `high_hz + transition_hz / 2 <= fs_out / 2`, otherwise `OutOfRange`. Works in pieces of 256 samples through two complex work buffers in the storage; per piece each kernel is called once (`rotate`, `fir_decimate`, `rotate`), never per sample, so the vector paths of the backends (`rotate` and `fir_decimate` are accelerated at every level since M13) are used and the block needs no SIMD code of its own. The piece is limited to the input that the filter can take with the room left in the output (`max_input_for_output`), so the first oscillator advances exactly by the consumed samples. **Chain order for SSB:** `ddc -> squelch -> agc -> ssb_demod`; the AGC defaults of M20 (attack 5 ms, release 0.2 s) suit SSB. **Tuning:** the tuning of an SSB station is done by the `ddc` frequency, the suppressed carrier must be at 0 Hz; a mistuning of `df` shifts the audio by `df`. **Convention:** amateur radio uses LSB below 10 MHz and USB above, marine and aviation HF voice is USB. `audio_decimation` above 1 also decimates the audio (6 gives 8 kHz from 48 kHz).
- **`biquad`:** `BiquadBlock<T>` (`C32`, `CQ15`; the real and the imaginary part are filtered alike) and `BiquadRealBlock<T>` (`f32`, `Q15`) around the kernels `biquad` and `biquad_real`. `response` is `lowpass`, `highpass`, `bandpass` or `notch`; low-pass and high-pass are Butterworth cascades of the even `order` 2 to 8 (`order / 2` sections, -3 dB at `cutoff_hz`), band-pass and notch are one cookbook section at the centre `cutoff_hz` with the quality `q` (default 0.7071) and need `order` 2. `cutoff_hz` (required) and `q` are runtime parameters: a change designs the cascade again and writes the new coefficients into the running filter with `set_coefs`, the history is kept and the output stays continuous; an invalid value is an error and changes nothing. The storage is `5 * sections` coefficients and `4 * sections` history values of the sample type (exact arena, no struct above 512 bytes). The filter is an IIR, so `timing()` reports no delay (the delay depends on the frequency, as for `deemphasis`). Q15 limit: the coefficients have 14 fractional bits and the rounding error of the feedback is amplified by `1 / |1 + a1 + a2|` at DC (above 50 dB for a low-pass or high-pass below about `fs / 500`), so `CQ15` and `Q15` reject a low-pass or high-pass with `cutoff_hz < fs / 500` with `OutOfRange` (at build time and as a runtime change); a design whose coefficients do not fit Q2.14 (a low-pass within 1e-6 of half the sample rate, where `b1` rounds to 2) is an error too. The Q15 filters can keep a small offset or limit cycle at silence (`docs/kernels.md`): the fourth-order Butterworth low-pass cycles with at most 1 LSB at 0.1 fs, 5 LSB at 0.05 fs and 54 LSB at 0.02 fs, so the guard `fs / 500`, which bounds the DC rounding amplification, still admits Q15 low cutoffs whose limit cycle exceeds 16 LSB (use `f32` there); the fourth-order high-pass at `fs / 160` keeps a constant DC rest of 321 LSB (about -40 dBFS) at silence, the approved Q15 bound of `nfm_receiver` (D22). A Q15 notch with Q 10 reaches about 34.9 dB at its centre (the Q2.14 rounding of its zeros), against more than 100 dB in `f32`; the thresholds are 34 dB for Q15 and 40 dB for `f32` (D23).
- **`halfband_decimate`:** `HalfbandDecimateBlock<T>` (`C32`, `CQ15`) and `HalfbandDecimateRealBlock<T>` (`f32`, `Q15`) around the kernels `halfband_decimate` and `halfband_decimate_real`: a Kaiser halfband low-pass with its cutoff at a quarter of the input rate, then decimation by 2 (output rate `fs / 2`). The tap count is `halfband_len(transition_hz, attenuation_db, fs)` (`4 k + 3`, at most `max_taps`, default 1023), `beta = kaiser_beta(attenuation_db)`; the taps are designed one at a time like the `fir` block (`halfband_taps` bit for bit), quantised to `f32` or Q15 with the zero taps staying exactly zero, so about half of the taps cost nothing: the cost is `(N + 3) / 4` multiplications per input sample (the `(N + 3) / 2` non-zero taps once per two input samples; a general decimating FIR needs `N / 2`). `transition_hz` (default `0.1 * fs`; it must be below `fs / 2`), `attenuation_db` (60) and `max_taps` are build parameters. The usable band after the decimation is `+-(fs / 4 - transition_hz / 2)` of the input rate; a signal above `fs / 4 + transition_hz / 2` is attenuated by at least `attenuation_db - 3` dB (it aliases into the band otherwise). The group delay is `(N - 1) / 2` input samples (`Delay::new(N - 1, 2)`), the first input sample after a reset produces the first output. The storage is `N` taps and `N - 1` history values (exact arena). **Cascades:** several `halfband_decimate` nodes in a row decimate by powers of two; give each stage the transition width it needs (the first stages can use a wide transition, because only the band that survives the later stages matters), for example 4.8 MS/s to 1.2 MS/s with two nodes in front of a `ddc`.
- **`real_to_iq`:** `RealToIqBlock<T>` (`f32` to `C32`, `Q15` to `CQ15`) around the kernel `real_to_complex`: the shift of the real input by `fs / 4` (multiplying sample `n` by `(-j)^n`, which costs no multiplication) and the halfband decimation in one step, same design, parameters and cost as `halfband_decimate`. **Tuning rule:** a real component at the frequency `f` (0 to `fs / 2`) appears in the complex output at `f - fs / 4` (the output rate is `fs / 2`); the output centre (0 Hz) is the input frequency `fs / 4`, and the mirror component at `-f` falls outside the pass band and is removed (at least `attenuation_db - 3` dB down). A station at the real frequency `f` is therefore tuned with a `ddc` frequency of `f - fs / 4` (a station at 1.8 MHz of a 4.8 MS/s input: `+600 kHz`). The counter `n` runs on across calls and restarts at `reset`; Q15 negation saturates (`-(-32768) = 32767`). A device that delivers a spectrum mirrored by `fs / 2` is handled by the sign of the `ddc` frequency.
- **`fm_stereo_demod`:** `FmStereoDemodBlock<T>` (`C32`, `CQ15`; output of the same type) is the stereo decoder for broadcast FM. It works through its input in pieces of `FM_STEREO_DEMOD_CHUNK` (256) samples; the result does not depend on the piece size. Processing order: (1) `quad_demod` with the gain of `fm_demod` gives the multiplex signal (MPX) `m`, (2) `biquad_real` with the RBJ band-pass `biquad_coefs(Bandpass, 19000, 5, fs)` (one section, zero phase at its centre) isolates the pilot `p`, (3) `stereo_pll(m, p)` with `pilot_hz` 19000, `natural_hz` = `pll_natural_hz` (20) and `pilot_level` 0.1 (CQ15: `0.1 * 2 * deviation_hz / fs`, because the discriminator reads `2 * df / fs` in units of pi) regenerates `sin(2 theta)` and forms `(m, m sin(2 theta))`, (4) `fir_decimate` (complex, factor `audio_decimation`) with the audio low-pass designed as in `fm_demod` (`DesignSpec` with the cutoff `audio_cutoff_hz`, a quarter of it as transition width, 60 dB, at most 1023 taps) gives `(S, D)`; both components pass the same filter, so sum and difference keep the same delay and gain (the separation depends on it), and with `audio_decimation` 1 the filter still runs, (5) `stereo_matrix` gives `(L, R) = (S + 2 D, S - 2 D)`, (6) the de-emphasis, a complex `biquad` with one section, the one-pole `[alpha, 0, 0, -(1 - alpha), 0]` with `alpha = 1 - exp(-1 / (fs_audio tau))`, acts on L and R after the matrix and is skipped for `deemphasis_us` 0 (applied to the MPX before the decoder, as `fm_demod` does, it would attenuate the subcarrier and destroy the separation). The output is stereo audio as complex samples, `re` left and `im` right (the convention of [pipeline.md](pipeline.md#stereo-audio)). The runtime parameter `stereo` (`auto` or `mono`, default `auto`) calls `set_enabled` on the loop state: `mono` zeroes the difference channel from the next sample while the loop keeps running, so `auto` restores stereo at once; the filter and the de-emphasis still hold the last stereo samples for about 5 ms, after that left and right are bit-identical. `pilot_locked()` returns `is_stereo()` of the loop (also in `mono`). The input rate must be at least 120 kHz (the MPX reaches 53 kHz and the products of step 3 reach 91 kHz), else `OutOfRange` (the pipeline reports `rate_unsupported`); the output rate must exceed `2 * audio_cutoff_hz + audio_cutoff_hz / 4`. Timing: ratio `1 / audio_decimation`, delay of `quad_demod` plus the FIR (the IIR stages count 0). `reset()` resets every state and keeps the runtime parameter. **Chain position:** behind a wide `ddc`: the FM signal of a stereo station is about 256 kHz wide, so the channel filter must pass about 2 x (75 + 53) kHz. With the values of `fm_receiver` (240 kHz, cutoff 100 kHz) the separation is limited to about 29 dB; `ddc` with decimation 5 (480 kHz) and cutoff 130 kHz, transition 30 kHz gives 43 dB (`examples/flows/fm_stereo_receiver.json`). **Measured** (E2E-STEREO, `crates/testing/sdrk-testkit/tests/e2e_stereo_m24.rs`): separation 43.1 and 44.9 dB (`f32` and Q15 alike), SINAD 67 and 65 dB, level within 0.06 dB, pilot residue below -106 dBFS, lock after 28 ms.
- **`quad_demod`:** `QuadDemodBlock<T>` (`C32` to `f32`, `CQ15` to `Q15`) around the kernel `quad_demod` with the gain of `fm_demod` (`C32`: `fs / (2 pi deviation_hz)`, the full deviation reads 1.0; `CQ15`: raw 32767, output unit phase per sample divided by pi, `deviation_hz` unused). One output per input; the first output after construction or `reset` reads 0. Needs no storage; the group delay is the half sample of the discriminator.
- **`deemphasis`:** `DeemphasisRealBlock<T>` (`f32`, `Q15`; no complex variant, a stereo de-emphasis is a `biquad`) around the kernel `deemphasis` with `DeemphasisState::new(tau_us * 1e-6, fs)`, `tau_us` 1 to 10000 (default 50). Needs no storage; the IIR group delay is reported as 0. The `f32` kernel of the SIMD backend is a tolerance kernel (see [kernels.md](kernels.md)).
- **`rotate`:** `RotateBlock<T>` (`C32`, `CQ15`) around the kernel `rotate` with `Nco::new(frequency_hz, fs)`: positive values shift the spectrum up (to bring a carrier at +f to 0 Hz use `-f`; `ddc` does this internally). `frequency_hz` is required, a runtime parameter, `|f| < fs / 2` (exclusive), and keeps the oscillator phase continuous. Needs no storage.
- **`decimate`:** `DecimateBlock<T>` (`C32`, `CQ15`) and `DecimateRealBlock<T>` (`f32`, `Q15`): the Kaiser low-pass designed at the input rate exactly as in `fir` (default cutoff `0.4 * fs / factor`, transition `0.1 * fs / factor`, 60 dB, `max_taps` 1023) and decimation by `factor` (1 to 1024). Complex signals use `fir_decimate`, real signals `resample_real` with `L = 1` (the kernel of the audio stage of `fm_demod` and `am_demod`, so a plan of those blocks stays bit-identical). The stop band `cutoff_hz + transition_hz / 2` must not exceed half the output rate (`OutOfRange`). Storage: the taps and the history (`taps - 1` complex samples, `taps` real samples). Group delay `(taps - 1) / 2` input samples. For a factor of 2 with a cheap filter use `halfband_decimate`.
- **`interpolate`:** `InterpolateBlock<T>` and `InterpolateRealBlock<T>` around `fir_interpolate` and `fir_interpolate_real`: taps designed at the *output* rate and multiplied by `factor`, exactly as in `duc`; cutoff (default `0.4 fs`) and transition (default `0.1 fs`) are relative to the *input* rate and `cutoff_hz + transition_hz / 2` must not exceed half the input rate. Storage: the taps and `ceil(taps / factor)` history samples. Group delay `(taps - 1) / (2 factor)` input samples.
- **`mag`:** `MagBlock<T>` (`C32` to `f32`, `CQ15` to `Q15`) around the kernel `mag`, the envelope detector of `am_demod`. No parameters, no state, no storage.
- **`delay`:** `DelayBlock<T>` (`C32`, `CQ15`) and `DelayRealBlock<T>` (`f32`, `Q15`): output `n` is input `n - samples`, the first `samples` outputs are zero. `samples` (0 to 65536, required, build time) elements of ring buffer come from the storage; the block copies in segments and calls no kernel. The group delay is exactly `samples`. It aligns two paths before a block with two inputs.
- **`add`, `mul`:** `AddBlock<T>`/`MulBlock<T>` (`C32`, `CQ15`) and `AddRealBlock<T>`/`MulRealBlock<T>` (`f32`, `Q15`) around the kernels `add`, `mul`, `add_real` and `mul_real`. The input is a `Two` port (lockstep): both slices have the same length, otherwise `LengthMismatch` and nothing changes; `Progress` counts pairs. The complex `mul` is the complex product. Q15 sums and products saturate (R7). No parameters, no storage, ratio 1, delay 0.
- **`make_complex`, `to_complex`:** `MakeComplexBlock<T>` (`Two<f32, f32>` or `Two<Q15, Q15>` to `C32`/`CQ15`, lockstep) and `ToComplexBlock<T>` (`f32`/`Q15` to `C32`/`CQ15` with imaginary part 0) around the kernel `interleave`. `real_part` and `imag_part` (`RealPartBlock<T>`, `ImagPartBlock<T>`, `C32`/`CQ15` to `f32`/`Q15`) around `deinterleave` with one output undo them bit for bit. The type parameter `T` of these four is the complex type.
- **`nco`:** `NcoBlock<T>` (clock is a `C32`/`CQ15` stream) and `NcoRealBlock<T>` (clock is an `f32`/`Q15` stream; `T` is the complex output type) around the kernel `oscillate`: one output sample `exp(j 2 pi k phase)` per input sample, the input values are ignored (the input is the clock, so a pipeline needs no source nodes). `frequency_hz` (runtime, required, strictly inside plus or minus half the rate; a change keeps the phase), `harmonic` (build, 1 to 8, the output is the harmonic `k` of the same phase, so harmonic 2 on the same clock is the square of harmonic 1) and `phase_turns` (build, 0 to 1, start phase; `reset` returns to it). `NcoConfig` is the config of both. No storage.
- **`stereo_pll`:** `StereoPllBlock<T>` (`Two<f32, f32>` or `Two<Q15, Q15>`, multiplex signal and pilot band-pass output, to `C32`/`CQ15`) around the kernel `stereo_pll` with the same loop state as `fm_stereo_demod`: with the same parameters and the same two inputs its output equals the internal loop of `fm_stereo_demod` bit for bit. `pilot_hz` (19000), `natural_hz` (20), `pilot_level` (0.1; for Q15 0.1 times the scale of the discriminator output), `stereo` (`auto`/`mono`, runtime), `pilot_locked()`. No storage.
- **`stereo_matrix`:** `StereoMatrixBlock<T>` (`C32`, `CQ15`) around the kernel `stereo_matrix`: `L = S + 2 D`, `R = S - 2 D` for sum in `re` and demodulated difference in `im`.
- **`resample`:** see below.
- **`spectrum`:** see below.

### Resample

`rate_out_hz` (required, a whole number of Hz) and the input rate define the ratio `L / M`.

- **Rational mode** (`tunable = false`, the default): `L` and `M` are the reduced ratio of the two rates (both at most 1024); the taps are designed at `L * rate_in` (default cutoff `0.45 * min(rate_in, rate_out)`, transition `0.1 * min(..)`, 60 dB) and multiplied by `L`; the block calls `resample` / `resample_real`. The ratio kind is `Constant`. Ratios like 44100/48000 (`147/160`) need thousands of taps at `L * rate_in`: set `max_taps` (the golden case uses 8192). A `ppm` other than 0 needs `tunable = true`; `set_param("ppm")` returns `ReadOnlyParameter`.
- **Tunable mode** (`tunable = true`): a polyphase bank of `P = 128` phases with `K` taps per phase (`K = ceil(taps / 128)` of the `kaiserord` design at `128 * rate_in`) and linear interpolation between neighbouring phases. The prototype low-pass of `128 K` taps is designed at `128 * rate_in` with gain 128; phase `p` uses `h[p + i * 128]`. The read position is a 32.32 fixed-point number of input samples, advanced by `step = round(2^32 * rate_in / rate_out / (1 + ppm * 1e-6))` per output. For an output at position `pos`: `n = pos >> 32`, `frac = pos & 0xFFFF_FFFF`, phase `p = frac >> 25` (top 7 bits), `mu = (frac >> 9) & 0xFFFF` (the next 16 bits, a Q16 weight); the output is `dot(x[n-K+1..=n], bank[p]) * (1 - mu) + dot(.., bank[p+1]) * mu`; the phase 128 is phase 0 applied one sample later (the output then waits for sample `n + 1`). `C32` interpolates in `f32`, `CQ15` computes `round_shift(a (65536 - mu) + b mu, 16)` in `i64` with saturation. `set_param("ppm")` (plus or minus 1000) recomputes the step and applies from the next output. The position is kept relative to the oldest sample still held, so it never overflows however long the stream runs; input between two windows (decimation by more than the work buffer) is skipped. The ratio kind is `Variable`; `timing().ratio` is the nominal `rate_out / rate_in`. With `ppm = +100`, a million input samples at 48 kHz give `1 000 100` outputs (plus or minus 1).

### Spectrum

For every frame the block takes `fft_size` consecutive samples starting at the frame position, multiplies them by a periodic window (`hann`, `blackman_harris` or `kaiser` with `kaiser_beta`), transforms them (`C32`: unnormalised; `CQ15`: block floating point), computes the power per bin, averages the power over `averaging` windows and converts it with `power_db` and the reference `(fft_size * coherent_gain)^2`, so that a full-scale complex tone at a bin centre reads 0 dBFS. The `CQ15` power is *not* `mag_sq` of Q15 (it saturates for large bins and rounds weak bins to zero) but `(re^2 + im^2) 2^(2 exponent - 30)` in `f32` with the exponent of the FFT. The frame is shifted so that the bins run from `-fs/2` to `+fs/2` with DC at index `fft_size / 2`. Consecutive windows start `interval_samples` apart (default `fft_size`; overlapping when smaller, samples are skipped when larger); both `averaging` and `interval_samples` are runtime parameters. `FrameMeta.first_sample` is the input index of the first sample of the first averaged window, counted since construction or the last `reset`.

## Clock drift

`DriftController` (`sdrk-blocks`, module `drift`, heap-free, f64 arithmetic with `libm::exp`) keeps the buffer between two clock domains (an SDR and a sound card, a file player and a transmitter) at a target fill by steering `ppm` of the tunable `resample` block. The buffer sits between the producer (the `resample` node at its output rate `fs`, corrected by `ppm`) and the consumer. A positive `ppm` makes the resampler produce more output per input, so a buffer that fills needs a lower `ppm`. The application calls `update(now_s, fill)` at roughly regular moments (for example in every audio callback) with a monotonic time in seconds and the fill level in samples.

`DriftConfig`: `rate_hz` (`fs`, greater than 0), `target` (samples, greater than 0), `time_constant_s` (default 30, 1 to 600), `max_ppm` (default 500, greater than 0 and at most 1000), `slew_ppm_per_s` (default 20, greater than 0), `smoothing_s` (default 1, greater than 0). `DriftController::new` returns `InvalidParameter` for a value outside its range (or NaN). Per call, with `dt = now_s - last` (a call with `dt <= 0` changes nothing and returns the current output; the first call only initialises the state and returns `initial_ppm`):

1. `e = (fill - target) / rate_hz` (seconds).
2. `a = 1 - exp(-dt / smoothing_s)`, `es = es + a (e - es)`.
3. `wn = 1 / time_constant_s`, `kp = 2 wn / 1e-6`, `ki = wn^2 / 1e-6` (a critically damped second-order loop for the plant `de/dt = 1e-6 ppm + drift`).
4. `integ = clamp(integ + ki es dt, +-max_ppm)` (anti-windup).
5. `u = clamp(initial_ppm - (kp es + integ), +-max_ppm)`.
6. The output moves towards `u` by at most `slew_ppm_per_s dt`.
7. The output is stored and returned.

`set_initial_ppm` sets a known offset (for example a measured device error), clamped to `max_ppm`; `reset` forgets the history and keeps the config and the initial ppm. This is a control loop, not a signal kernel: it has no SIMD variant and no golden vectors.

## Tests

- `tests/two_port_blocks.rs` (the blocks of M30: equality with the kernels in one piece and in chunks for all four sample types, lockstep `LengthMismatch` of the `Two` inputs, `make_complex` to `real_part`/`imag_part` bit for bit with NaN, infinities and negative zero, `nco` against `oscillate`, frequency change keeps the phase, harmonic 2 is the square of harmonic 1, start phase and reset, parameter validation, `stereo_pll` against the kernel and, followed by the rest of the stereo decoder, against `fm_stereo_demod` bit for bit, zero storage).
- Suite cases of M30: `block_nco_phase_lock` (10^6 samples, full tier) and `block_make_complex_roundtrip` (quick).
- `tests/biquad.rs` (the block `biquad`: a 4th-order low-pass at 2 kHz of 48 kHz passes 1 kHz with less than 0.1 dB and stops 8 kHz by at least 48 dB in `f32` and Q15, both parts of the complex blocks, the notch at 1 kHz with q 10 (40 dB in `f32`, 34 dB in Q15, D23), runtime `cutoff_hz` and `q` against the kernel with `set_coefs`, invalid values, the Q15 cutoff limit at `fs / 500`, coefficients that do not fit Q2.14, odd orders, `order` other than 2 for band-pass and notch, 20-plan invariance for all four sample types, reset, NaN and infinity, the description); `construction.rs` adds the exact arena and the 512-byte limit, `describe.rs` the description checks.
- `tests/halfband_decimate.rs` (the blocks `halfband_decimate` and `real_to_iq` and the real `convert`: the f32 taps are the taps of `halfband_taps` bit for bit, the Q15 zero taps stay zero, the group delay against an impulse within 0.5 sample, a stop-band tone attenuated by at least `attenuation_db - 3` dB (complex and real, `f32` and Q15), the pass band flat, `real_to_iq` maps `f` to `f - fs/4` and removes the mirror, 20-plan invariance for all sample types, `reset` restarts `n mod 4`, exact arena, 512-byte limit, `Send`/`Sync`, parameters, invalid designs, `convert` with `rs16` equals `convert_real`).
- `tests/primitive_blocks.rs` (the seven blocks `quad_demod`, `deemphasis`, `rotate`, `decimate`, `interpolate`, `mag` and `delay`: every block equals the direct kernel call bit for bit for each sample type, whole and in 20 chunk plans (seeds 455 to 461; the chunked `f32` `deemphasis` is compared by SNR, D225); `rotate` then `decimate` equals `ddc`, `interpolate` then `rotate` equals `duc` and `mag` equals the first stage of `am_demod`, all bit for bit; the runtime `frequency_hz` of `rotate` with continuous phase; exact storage in an arena, small structs, timing and rates, validation errors).
- Golden cases (`design/`, `dot/`, `blocks/`): blocks are driven in one piece and in random chunks; results must be identical and match SciPy/NumPy (`ddc` 130 dB, FM demodulator 133 dB, rational resampler 139 dB, tuned resampler 140 dB (`C32`) and bit-exact (`CQ15`), spectrum within 0.0002 dB (`C32`) and 0.0013 dB (`CQ15`)).
- `tests/describe.rs`, `construction.rs`, `params.rs`, `behaviour.rs`, `squelch.rs` (gate samples, hysteresis band, runtime parameters, invariance), `agc.rs` (every runtime parameter, kernel equality, 20-plan invariance), `am_demod.rs` (kernel chain, envelope level, 20-plan invariance), `ssb_demod.rs` (kernel chain, tone level and frequency in both sidebands, opposite-sideband rejection, runtime switch, validation, 20-plan invariance), `timing.rs`, `map_index.rs`, `ppm.rs`, `functional.rs`, `design.rs`, `storage.rs` in `crates/blocks/sdrk-blocks`: the rules above, exact storage, parameters, chunk invariance and `reset` of every block and sample type, extreme inputs (silence, full scale, NaN, infinity), impulse delays, output counts, pass bands and stop bands, DUC-DDC and FM modulator-demodulator loops, spectrum levels.
- Suite cases (no heap, `ArenaStorage` on the suite arena): `block_ddc_vs_chain`, `block_fm_demod_sinad` (the receive chain of M08 rebuilt from blocks, thresholds 40/35 dB; it reaches about 98/86 dB because the designed filters are better than the Hann filters of the kernel chain), `block_spectrum_tone_level` (0 dBFS plus or minus 0.05 dB for `C32`, plus or minus 0.3 dB for `CQ15`), `block_resample_ppm_count`.
