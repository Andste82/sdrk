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

`WriteReporter` prints it to any `core::fmt::Write`, `StdReporter` (feature `std`) to stdout. Registered by M03: `rng_known_answer`, `gen_tone_frequency`, `encode_decode_roundtrip_all_formats`, `scale_fallback_vs_reference` (quick) and `scale_block_invariance` (full). New kernels add their cases to `cases.rs`. On the host, `cargo test -p sdrk-testkit --features std` runs the whole suite in `tests/suite.rs`.
