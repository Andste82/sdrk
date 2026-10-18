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
