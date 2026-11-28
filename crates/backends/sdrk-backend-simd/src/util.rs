//! Small helpers shared by the kernels.

use sdrk_core::{Error, Result};

/// `Ok` when both lengths are equal, else `LengthMismatch { expected, got }`.
#[inline(always)]
pub(crate) fn check_len(expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::LengthMismatch { expected, got })
    }
}

/// Runs `body` on every pair of whole blocks of `in_step` elements of `input` and `out_step`
/// elements of `out`; returns the number of pairs. The plain chunk iterators zip to one counted
/// loop (zipping `&mut ChunksExact` would test both ends in every iteration).
#[inline(always)]
pub(crate) fn blocks<T, U>(
    input: &[T],
    out: &mut [U],
    in_step: usize,
    out_step: usize,
    mut body: impl FnMut(&[T], &mut [U]),
) -> usize {
    let n = (input.len() / in_step).min(out.len() / out_step);
    let (input, _) = input.split_at(n * in_step);
    let (out, _) = out.split_at_mut(n * out_step);
    for (i, o) in input
        .chunks_exact(in_step)
        .zip(out.chunks_exact_mut(out_step))
    {
        body(i, o);
    }
    n
}

/// The error for a violated internal invariant.
#[cold]
pub(crate) fn internal(what: &'static str) -> Error {
    Error::Internal(what)
}

/// `pmulhrsw` (`_mm_mulhrs_epi16`, SSSE3) of the `__m128i` values `$a` and `$b` in SSE2
/// instructions, bit for bit: lane by lane `((a b >> 14) + 1) >> 1` in 16 bits, so
/// `-32768 * -32768` wraps to `-32768` as with `pmulhrsw` (decision D3008). With `hi` and `lo`
/// the signed high and the unsigned low half of the product `p = 65536 hi + lo`,
/// `p >> 14 = 4 hi + (lo >> 14)`, and as `4 hi` is even,
/// `((p >> 14) + 1) >> 1 = 2 hi + (((lo >> 14) + 1) >> 1)`; the last term is `pavgw` of
/// `lo >> 14` (0 to 3) with zero. Six instructions: `pmulhw`, `pmullw`, `psrlw`, `pavgw` and two
/// `paddw`. A macro, so that the intrinsics are called inside the `fearless_simd::kernel!`
/// function of the level (where they are safe).
#[cfg(target_arch = "x86_64")]
macro_rules! mulhrs_sse2 {
    ($a:expr, $b:expr $(,)?) => {{
        let (a, b): (core::arch::x86_64::__m128i, core::arch::x86_64::__m128i) = ($a, $b);
        let hi = core::arch::x86_64::_mm_mulhi_epi16(a, b);
        let lo =
            core::arch::x86_64::_mm_srli_epi16::<14>(core::arch::x86_64::_mm_mullo_epi16(a, b));
        core::arch::x86_64::_mm_add_epi16(
            core::arch::x86_64::_mm_add_epi16(hi, hi),
            core::arch::x86_64::_mm_avg_epu16(lo, core::arch::x86_64::_mm_setzero_si128()),
        )
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use mulhrs_sse2;

/// `q15_mul` of the `__m128i` values `$a` and `$b` in SSE2 instructions, bit for bit: the
/// sequence of [`mulhrs_sse2!`] with saturating additions (`paddsw`), so that the one product
/// outside the 16-bit range, `-32768 * -32768`, gives 32767 instead of wrapping (decision
/// D3008). `2 hi` saturates only for that product (`hi = 16384`, the low half and its term 0);
/// every other result `2 hi + t` is in range, so the saturating additions change nothing else.
#[cfg(target_arch = "x86_64")]
macro_rules! q15_mul_sse2 {
    ($a:expr, $b:expr $(,)?) => {{
        let (a, b): (core::arch::x86_64::__m128i, core::arch::x86_64::__m128i) = ($a, $b);
        let hi = core::arch::x86_64::_mm_mulhi_epi16(a, b);
        let lo =
            core::arch::x86_64::_mm_srli_epi16::<14>(core::arch::x86_64::_mm_mullo_epi16(a, b));
        core::arch::x86_64::_mm_adds_epi16(
            core::arch::x86_64::_mm_adds_epi16(hi, hi),
            core::arch::x86_64::_mm_avg_epu16(lo, core::arch::x86_64::_mm_setzero_si128()),
        )
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use q15_mul_sse2;

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use core::arch::x86_64::{_mm_mulhrs_epi16, _mm_set1_epi16, _mm_setr_epi16};
    use fearless_simd::i16x8;
    use fearless_simd::prelude::*;

    fearless_simd::kernel!(
        /// [`mulhrs_sse2!`] against `pmulhrsw` for every `a` and a set of `b` (stepping through
        /// the range with a stride prime to 2^16, and the extremes).
        fn compare(t: Sse4_2) {
            let bs = (0..=65_535i32)
                .step_by(251)
                .map(|v| (v - 32_768) as i16)
                .chain([i16::MIN, -1, 0, 1, i16::MAX]);
            for b in bs {
                for a in (i16::MIN..=i16::MAX).step_by(8) {
                    let x = _mm_setr_epi16(a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7);
                    let y = _mm_set1_epi16(b);
                    let got = i16x8::simd_from(t, mulhrs_sse2!(x, y));
                    let want = i16x8::simd_from(t, _mm_mulhrs_epi16(x, y));
                    assert_eq!(got.as_slice(), want.as_slice(), "{a} {b}");
                }
            }
        }
    );

    fearless_simd::kernel!(
        /// [`q15_mul_sse2!`] against `q15_mul` of `sdrk-core` for every `a` and the `b` of
        /// [`compare`].
        fn compare_q15_mul(t: Sse2) {
            let bs = (0..=65_535i32)
                .step_by(251)
                .map(|v| (v - 32_768) as i16)
                .chain([i16::MIN, -1, 0, 1, i16::MAX]);
            for b in bs {
                for a in (i16::MIN..=i16::MAX).step_by(8) {
                    let x = _mm_setr_epi16(a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6, a + 7);
                    let got = i16x8::simd_from(t, q15_mul_sse2!(x, _mm_set1_epi16(b)));
                    let want: [i16; 8] = core::array::from_fn(|k| {
                        sdrk_core::num::q15_mul(sdrk_core::Q15(a + k as i16), sdrk_core::Q15(b)).0
                    });
                    assert_eq!(got.as_slice(), want.as_slice(), "{a} {b}");
                }
            }
        }
    );

    #[test]
    fn q15_mul_sse2_is_q15_mul() {
        if let Some(t) = crate::level::current().as_sse2() {
            compare_q15_mul(t);
        }
    }

    #[test]
    fn mulhrs_sse2_is_pmulhrsw() {
        if let Some(t) = crate::level::current().as_sse4_2() {
            compare(t);
        }
    }
}
