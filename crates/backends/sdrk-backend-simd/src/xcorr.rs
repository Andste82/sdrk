//! `xcorr_fft` `C32` (M50 §5): the overlap-save fast convolution.
//!
//! The driver is the one of the fallback (`sdrk_core::kernels::xcorr::xcorr_fft_with`: the
//! block boundaries every `B` input samples and the history copy), compiled at the dispatched
//! level, as `channelize` does (D918). Per block the window `[history | B new samples]` is copied
//! into the scratch, transformed with the `C32` FFT of this backend at the same level
//! (`fft::forward_at`), multiplied bin by bin with the tap spectrum `H` ([`spectral_product`]),
//! transformed back with the level's inverse (`fft::inverse_at`, scaled by `1 / N`) and its
//! last `B` values are the outputs. No second dispatch per transform.
//!
//! **The spectral product** uses the fixed formula of the contract on whole vectors: the four
//! products `wr hr`, `wi hi`, `wr hi`, `wi hr`, then `re = wr hr - wi hi` and
//! `im = wr hi + wi hr`, every operation one rounded `f32` operation, **no FMA** even where the
//! level has it (unlike `mul`, D326). So every non-NaN result has the bits of
//! `sdrk_core::kernels::xcorr::spectral_product_one`; a NaN result may differ in sign and payload
//! (D296). The samples are split into a vector of real parts and one of imaginary parts and
//! joined again with the `Split` of the level (the in-lane shuffles at AVX2 and AVX-512,
//! `ld2`/`st2` on NEON, the deinterleave of `fearless_simd` elsewhere); the order of the samples
//! in the lanes does not matter, since the product is element-wise and the store undoes the
//! load. The samples after the last whole vector take the scalar formula.
//!
//! The class is *tolerance* (`docs/kernels.md`): the copies and the product are exact, only the
//! FFT of the level makes the output differ from the fallback (shared conventions §D.3). On one
//! level the output is bit-identical for every chunking.
//!
//! The public [`spectral_product`] applies the vector product to slices at the level of the
//! running build; it exists for the tests of the helper (`tests/xcorr.rs`) and for callers
//! outside the kernel.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::xcorr::{
    BlockArgs, XcorrFft, XcorrFftState, spectral_product_one, xcorr_fft_with,
};
use sdrk_core::timing::Progress;
use sdrk_core::{C32, Result};

use crate::Simd;
use crate::fft::{forward_at, inverse_at};
use crate::level::dispatch;
use crate::mix::{Portable, Split};
use crate::util::{check_len, internal};

/// The error of a violated length invariant of a block (the state checks the lengths).
#[cold]
fn bad_len() -> sdrk_core::Error {
    internal("xcorr_fft block")
}

/// `w[k] = w[k] h[k]` on the interleaved floats of `w` and `h` (equal lengths) with the
/// [`Split`] `split` of the level: whole vectors of `S::f32s::LEN` samples with the vector
/// formula, the rest with the scalar formula.
#[inline(always)]
fn product_split<S: SimdToken, L: Split<S>>(simd: S, split: L, w: &mut [f32], h: &[f32]) {
    let step = 2 * S::f32s::LEN;
    let whole = w.len().min(h.len()) / step * step;
    let (w_vec, w_rest) = w.split_at_mut(whole);
    let (h_vec, h_rest) = h.split_at(whole);
    for (w, h) in w_vec.chunks_exact_mut(step).zip(h_vec.chunks_exact(step)) {
        let (wr, wi) = split.load(simd, w);
        let (hr, hi) = split.load(simd, h);
        // The four products first, then one subtraction and one addition: no FMA.
        let rr = wr * hr;
        let ii = wi * hi;
        let ri = wr * hi;
        let ir = wi * hr;
        split.store(simd, rr - ii, ri + ir, w);
    }
    for (w, h) in w_rest.chunks_exact_mut(2).zip(h_rest.chunks_exact(2)) {
        if let ([wr, wi], [hr, hi]) = (w, h) {
            let y = spectral_product_one(C32::new(*wr, *wi), C32::new(*hr, *hi));
            (*wr, *wi) = (y.re, y.im);
        }
    }
}

/// Copies `src` to `dst` (equal lengths) with whole vectors of the level and the rest sample by
/// sample: a call of `memcpy` would cost a call per copy and, for the copies of thousands of
/// samples, a `rep movsb` in glibc, which the counters of the cycle simulation count once per
/// byte (`docs/performance.md`). Exact (plain loads and stores).
#[inline(always)]
fn copy_at<S: SimdToken>(simd: S, dst: &mut [C32], src: &[C32]) {
    let d: &mut [f32] = bytemuck::cast_slice_mut(dst);
    let s: &[f32] = bytemuck::cast_slice(src);
    let step = 2 * S::f32s::LEN;
    let whole = d.len().min(s.len()) / step * step;
    let (d_vec, d_rest) = d.split_at_mut(whole);
    let (s_vec, s_rest) = s.split_at(whole);
    for (d, s) in d_vec.chunks_exact_mut(step).zip(s_vec.chunks_exact(step)) {
        let (d0, d1) = d.split_at_mut(S::f32s::LEN);
        let (s0, s1) = s.split_at(S::f32s::LEN);
        S::f32s::from_slice(simd, s0).store_slice(d0);
        S::f32s::from_slice(simd, s1).store_slice(d1);
    }
    for (d, s) in d_rest.iter_mut().zip(s_rest) {
        *d = *s;
    }
}

/// `w[k] = w[k] h[k]` at the level of `simd` with the [`Split`] of that level (equal lengths).
#[inline(always)]
fn product_at<S: SimdToken>(simd: S, w: &mut [C32], h: &[C32]) {
    let w: &mut [f32] = bytemuck::cast_slice_mut(w);
    let h: &[f32] = bytemuck::cast_slice(h);
    let level = simd.level();
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = level.as_avx512() {
            return product_split(t, crate::mix::x86::InLane, w, h);
        }
        if let Some(t) = level.as_avx2() {
            return product_split(t, crate::mix::x86::InLane, w, h);
        }
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = level.as_neon() {
        return product_split(t, crate::mix::neon::Ld2, w, h);
    }
    let _ = level;
    product_split(simd, Portable, w, h);
}

/// The spectral product of `xcorr_fft` at the level of the running build: `w[k] = w[k] h[k]`
/// for every bin with the fixed formula of the contract (four products, then
/// `re = wr hr - wi hi` and `im = wr hi + wi hr`, no FMA). Every non-NaN result has the bits of
/// `sdrk_core::kernels::xcorr::spectral_product_one`; a NaN result may have another sign and
/// payload (D296).
///
/// # Errors
/// `LengthMismatch` if the lengths differ.
///
/// ```
/// use sdrk_backend_simd::xcorr::spectral_product;
/// use sdrk_core::C32;
/// let mut w = [C32::new(1.0, 2.0); 5];
/// spectral_product(&mut w, &[C32::new(3.0, -1.0); 5])?;
/// assert!(w.iter().all(|v| *v == C32::new(5.0, 5.0)));
/// # Ok::<(), sdrk_core::Error>(())
/// ```
pub fn spectral_product(w: &mut [C32], h: &[C32]) -> Result<()> {
    check_len(w.len(), h.len())?;
    dispatch!(simd => product_at(simd, w, h));
    Ok(())
}

/// One overlap-save block at the level of `simd` (the definition of
/// `sdrk_core::kernels::xcorr::overlap_save_block` with the FFT and the product of the level).
#[inline(always)]
fn block_at<S: SimdToken, Tw: AsRef<[C32]>>(
    simd: S,
    (history, x, spectrum, scratch, plan, y): BlockArgs<'_, C32, Tw>,
) -> Result<()> {
    let keep = history.len();
    if keep + x.len() != scratch.len() || y.len() != x.len() || spectrum.len() != scratch.len() {
        return Err(bad_len());
    }
    let (head, tail) = scratch.split_at_mut(keep);
    copy_at(simd, head, history);
    copy_at(simd, tail, x);
    forward_at(simd, plan, scratch)?;
    product_at(simd, scratch, spectrum);
    inverse_at(simd, plan, scratch)?;
    copy_at(simd, y, scratch.get(keep..).ok_or_else(bad_len)?);
    Ok(())
}

/// The fast convolution at the level of `simd`.
#[inline(always)]
fn xcorr_level<S, Sp, Sh, Sc, Tw>(
    simd: S,
    state: &mut XcorrFftState<C32, Sp, Sh, Sc, Tw>,
    input: &[C32],
    output: &mut [C32],
) -> Result<Progress>
where
    S: SimdToken,
    Sp: AsRef<[C32]>,
    Sh: AsRef<[C32]> + AsMut<[C32]>,
    Sc: AsRef<[C32]> + AsMut<[C32]>,
    Tw: AsRef<[C32]>,
{
    xcorr_fft_with(
        state,
        input,
        output,
        #[inline(always)]
        |args: BlockArgs<'_, C32, Tw>| block_at(simd, args),
    )
}

impl XcorrFft<C32> for Simd {
    fn xcorr_fft<Sp, Sh, Sc, Tw>(
        &self,
        state: &mut XcorrFftState<C32, Sp, Sh, Sc, Tw>,
        input: &[C32],
        output: &mut [C32],
    ) -> Result<Progress>
    where
        Sp: AsRef<[C32]>,
        Sh: AsRef<[C32]> + AsMut<[C32]>,
        Sc: AsRef<[C32]> + AsMut<[C32]>,
        Tw: AsRef<[C32]>,
    {
        dispatch!(simd => xcorr_level(simd, state, input, output))
    }
}
