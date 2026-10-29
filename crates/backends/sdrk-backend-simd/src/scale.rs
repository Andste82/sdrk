//! `scale` `CQ15`: `q15_mul(x, gain)` of every component, bit-exact with the fallback.
//!
//! The rounding high multiplication of the levels is `q15_mul` itself: `pmulhrsw` (SSSE3, so
//! SSE4.2, AVX2 and AVX-512) and `i16x8.q15mulr_sat_s` compute `(x * g + 2^14) >> 15`, and
//! `sqrdmulh` (NEON) `(2 x g + 2^15) >> 16`, the same value. The only result outside the 16-bit
//! range is `-1 * -1 = 32768`: NEON and WASM saturate it to 32767 like `q15_mul`, `pmulhrsw`
//! wraps it to -32768, so on x86 a gain of -32768 takes the saturating negation instead
//! (`q15_mul(x, -32768)` is `-x` limited to 32767). SSE2 has no rounding high multiplication
//! and runs the fallback.
//!
//! `C32`: the multiplication of every float by the gain in generic code, bit-exact with the
//! fallback; the backend has it so that the build with run-time detection runs it at the level
//! of the CPU (the fallback is compiled for the baseline of the target there).

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::basic::Scale;
use sdrk_core::{C32, CQ15, Fallback, Q15, Result};

use crate::Simd;
use crate::level::dispatch_fn;
use crate::util::check_len;

/// The level-specific path for the current level, if there is one: returns the number of
/// leading values done.
#[inline(always)]
fn scale_q15_level<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16], gain: i16) -> usize {
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return x86::scale_avx512(t, input, out, gain);
        }
        if let Some(t) = level.as_avx2() {
            return x86::scale_avx2(t, input, out, gain);
        }
        if let Some(t) = level.as_sse4_2() {
            return x86::scale_sse4_2(t, input, out, gain);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return neon::scale(t, input, out, gain);
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = level.as_wasm_simd128() {
        return wasm::scale(t, input, out, gain);
    }
    let _ = (level, input, out, gain);
    0
}

#[inline(always)]
fn scale_q15<S: SimdToken>(simd: S, input: &[i16], out: &mut [i16], gain: i16, done: &mut usize) {
    *done = scale_q15_level(simd, input, out, gain);
}

/// `x * gain` of every float, bit-exact with the fallback (one multiplication per component,
/// in the order of the samples).
#[inline(always)]
fn scale_f32<S: SimdToken>(simd: S, input: &[f32], out: &mut [f32], gain: f32) {
    let l = S::f32s::LEN;
    let g = S::f32s::splat(simd, gain);
    let n = input.len().min(out.len()) / (8 * l) * (8 * l);
    let (ib, ir) = input.split_at(n);
    let (ob, or) = out.split_at_mut(n);
    for (i, o) in ib.chunks_exact(8 * l).zip(ob.chunks_exact_mut(8 * l)) {
        for (i, o) in i.chunks_exact(l).zip(o.chunks_exact_mut(l)) {
            (S::f32s::from_slice(simd, i) * g).store_slice(o);
        }
    }
    for (i, o) in ir.iter().zip(or) {
        *o = *i * gain;
    }
}

// The entries, one function per level (see `dispatch_fn!`).
dispatch_fn!(fn run_f32(input: &[f32], out: &mut [f32], gain: f32) => scale_f32);
dispatch_fn!(fn run_q15(input: &[i16], out: &mut [i16], gain: i16, done: &mut usize) => scale_q15);

impl Scale<C32> for Simd {
    fn scale(&self, input: &[C32], output: &mut [C32], gain: f32) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[f32] = bytemuck::cast_slice(input);
        let y: &mut [f32] = bytemuck::cast_slice_mut(output);
        run_f32(x, y, gain);
        Ok(())
    }
}

impl Scale<CQ15> for Simd {
    fn scale(&self, input: &[CQ15], output: &mut [CQ15], gain: Q15) -> Result<()> {
        check_len(input.len(), output.len())?;
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        let mut done = 0;
        run_q15(x, y, gain.0, &mut done);
        // The rest (and everything on a level without a path), whole samples.
        let s = done / 2;
        match (input.get(s..), output.get_mut(s..)) {
            (Some(i), Some(o)) if !i.is_empty() => Fallback.scale(i, o, gain),
            _ => Ok(()),
        }
    }
}

/// `$body` with `$i` and `$o` the blocks of `$v` values of `$input` and `$out`; evaluates to the
/// number of values done (a multiple of `2 $v`, whole samples).
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
macro_rules! blocks {
    ($input:expr, $out:expr, $v:literal, |$i:ident, $o:ident| $body:block) => {{
        let n = $input.len().min($out.len()) / $v * $v;
        let (input, _) = $input.split_at(n);
        let (out, _) = $out.split_at_mut(n);
        // One closure for both loops (see `pairs!` in `dot.rs`).
        let body = |$i: &[i16; $v], $o: &mut [i16; $v]| $body;
        // Groups of eight blocks (unrolled), then single blocks.
        let g = n / (8 * $v) * (8 * $v);
        let (ig, ir) = input.split_at(g);
        let (og, or) = out.split_at_mut(g);
        let (ig, _) = ig.as_chunks::<{ 8 * $v }>();
        let (og, _) = og.as_chunks_mut::<{ 8 * $v }>();
        for (iq, oq) in ig.iter().zip(og) {
            let (iq, _) = iq.as_chunks::<$v>();
            let (oq, _) = oq.as_chunks_mut::<$v>();
            for (i, o) in iq.iter().zip(oq) {
                body(i, o);
            }
        }
        let (ir, _) = ir.as_chunks::<$v>();
        let (or, _) = or.as_chunks_mut::<$v>();
        for (i, o) in ir.iter().zip(or) {
            body(i, o);
        }
        n
    }};
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::x86_64::{
        _mm_mulhrs_epi16, _mm_set1_epi16, _mm_setzero_si128, _mm_subs_epi16, _mm256_mulhrs_epi16,
        _mm256_set1_epi16, _mm256_setzero_si256, _mm256_subs_epi16, _mm512_mulhrs_epi16,
        _mm512_set1_epi16, _mm512_setzero_si512, _mm512_subs_epi16,
    };
    use fearless_simd::prelude::*;
    use fearless_simd::{i16x8, i16x16, i16x32};

    fearless_simd::kernel!(
        pub(super) fn scale_sse4_2(t: Sse4_2, input: &[i16], out: &mut [i16], gain: i16) -> usize {
            if gain == i16::MIN {
                let zero = _mm_setzero_si128();
                blocks!(input, out, 8, |i, o| {
                    let x = i16x8::from_slice(t, i).into();
                    i16x8::simd_from(t, _mm_subs_epi16(zero, x)).store_array(o);
                })
            } else {
                let g = _mm_set1_epi16(gain);
                blocks!(input, out, 8, |i, o| {
                    let x = i16x8::from_slice(t, i).into();
                    i16x8::simd_from(t, _mm_mulhrs_epi16(x, g)).store_array(o);
                })
            }
        }
    );

    fearless_simd::kernel!(
        pub(super) fn scale_avx2(t: Avx2, input: &[i16], out: &mut [i16], gain: i16) -> usize {
            if gain == i16::MIN {
                let zero = _mm256_setzero_si256();
                blocks!(input, out, 16, |i, o| {
                    let x = i16x16::from_slice(t, i).into();
                    i16x16::simd_from(t, _mm256_subs_epi16(zero, x)).store_array(o);
                })
            } else {
                let g = _mm256_set1_epi16(gain);
                blocks!(input, out, 16, |i, o| {
                    let x = i16x16::from_slice(t, i).into();
                    i16x16::simd_from(t, _mm256_mulhrs_epi16(x, g)).store_array(o);
                })
            }
        }
    );

    fearless_simd::kernel!(
        pub(super) fn scale_avx512(t: Avx512, input: &[i16], out: &mut [i16], gain: i16) -> usize {
            if gain == i16::MIN {
                let zero = _mm512_setzero_si512();
                blocks!(input, out, 32, |i, o| {
                    let x = i16x32::from_slice(t, i).into();
                    i16x32::simd_from(t, _mm512_subs_epi16(zero, x)).store_array(o);
                })
            } else {
                let g = _mm512_set1_epi16(gain);
                blocks!(input, out, 32, |i, o| {
                    let x = i16x32::from_slice(t, i).into();
                    i16x32::simd_from(t, _mm512_mulhrs_epi16(x, g)).store_array(o);
                })
            }
        }
    );
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::{vdupq_n_s16, vqrdmulhq_s16};
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    fearless_simd::kernel!(
        pub(super) fn scale(t: Neon, input: &[i16], out: &mut [i16], gain: i16) -> usize {
            let g = vdupq_n_s16(gain);
            blocks!(input, out, 8, |i, o| {
                let x = i16x8::from_slice(t, i).into();
                i16x8::simd_from(t, vqrdmulhq_s16(x, g)).store_array(o);
            })
        }
    );
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use core::arch::wasm32::{i16x8_q15mulr_sat, i16x8_splat};
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    fearless_simd::kernel!(
        pub(super) fn scale(t: WasmSimd128, input: &[i16], out: &mut [i16], gain: i16) -> usize {
            let g = i16x8_splat(gain);
            blocks!(input, out, 8, |i, o| {
                let x = i16x8::from_slice(t, i).into();
                i16x8::simd_from(t, i16x8_q15mulr_sat(x, g)).store_array(o);
            })
        }
    );
}
