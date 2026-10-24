//! The complex FFT for `C32` and `CQ15`.
//!
//! # `C32`
//!
//! A decimation-in-time transform in place without a separate bit-reversal pass: one pass over
//! tiles does the bit reversal, the first stages and the conversion to blocks in split form, then
//! radix-4 passes (three complex products per butterfly) finish the transform, the last one
//! storing interleaved samples. The twiddles come from static per-span tables (feature
//! `fft-tables`) or from the plan. See `c32` for the details.
//!
//! The result is within the contract (110 dB) of the exact transform, not bit-identical with the
//! fallback: the stages are fused, FMA is used where the level has it, and the order of the
//! operations differs.
//!
//! # `CQ15`
//!
//! Radix-4 stages in `i16` lanes per level (see `q15`), within the SNR contract of
//! docs/kernels.md but not bit-identical with the fallback. Small sizes and levels without a
//! path use the fallback's radix-2 arithmetic (the twiddle products and sums of one butterfly
//! fit `i32` for twiddles made by `FftPlan::new`) and are bit-identical there.
//!
//! # `rfft`, `irfft`
//!
//! Around the `C32` FFT of `N / 2` points of this backend: `rfft` forms 4 bins and their 4
//! mirrors per vector at SSE2, SSE4.2, NEON and SIMD128 (the fallback's loop, compiled at the
//! level, at AVX2 and AVX-512), `irfft` is the fallback's code
//! (`sdrk_core::kernels::fft::irfft_with`) compiled at the dispatched level (see `real`).

mod c32;
mod q15;
mod real;
