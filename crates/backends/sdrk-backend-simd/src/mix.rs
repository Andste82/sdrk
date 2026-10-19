//! `rotate`: frequency shift with a numerically controlled oscillator.
//!
//! `C32`: the oscillator values come from two short polynomials instead of the cosine table, so
//! no table lookups are needed in the vector loop (a table needs one load per lane, which costs
//! more instructions than the polynomials). The result meets the contract of the kernel (120 dB
//! against the exact rotation; the polynomials reach about 140 dB) but is not bit-identical with
//! the fallback. FMA is used where the level has it. The samples are processed deinterleaved,
//! one vector of oscillator values per vector of real parts.
//!
//! `CQ15`: the table lookup and the integer arithmetic of the fallback, vectorised, bit-exact.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::mix::{Nco, Rotate};
use sdrk_core::{C32, CQ15, Fallback, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::table::cos_sin_q15;
use crate::util::check_len;

/// Coefficients of `sin(pi/2 x) = x (S0 + S1 z + ... + S4 z^4)`, `z = x^2`, on `[0, 1]`
/// (interpolating polynomial at the Chebyshev nodes, maximum error 7e-9).
const SIN: [f32; 5] = [
    core::f32::consts::FRAC_PI_2,
    -0.645_963_7,
    0.079_689_92,
    -0.004_674_143_8,
    0.000_151_671_7,
];
/// Coefficients of `cos(pi/2 x) = C0 + C1 z + ... + C5 z^5` on `[0, 1]` (maximum error 2e-10).
const COS: [f32; 6] = [
    0.999_999_96,
    -1.233_700_5,
    0.253_669_32,
    -0.020_862_687,
    0.000_917_723,
    -0.000_023_824_15,
];

/// `(cos, sin)` of the phases `p` (units of `2 pi / 2^32`), accurate to about 1e-7, without a
/// table: the quadrant comes from the top two bits, the rest is a fraction of a quarter turn
/// that two short polynomials evaluate. Any `p` gives the same result whatever the other lanes
/// hold, so the output does not depend on how a signal is split into blocks.
#[inline(always)]
fn cos_sin_poly<S: SimdToken>(simd: S, p: S::u32s) -> (S::f32s, S::f32s) {
    let frac: S::i32s = (p & 0x3FFF_FFFFu32).bitcast();
    let x = frac.to_float::<S::f32s>() * (1.0 / 1_073_741_824.0);
    let z = x * x;
    let mut sp = S::f32s::splat(simd, SIN[4]);
    for c in [SIN[3], SIN[2], SIN[1], SIN[0]] {
        sp = sp.mul_add(z, S::f32s::splat(simd, c));
    }
    let mut cp = S::f32s::splat(simd, COS[5]);
    for c in [COS[4], COS[3], COS[2], COS[1], COS[0]] {
        cp = cp.mul_add(z, S::f32s::splat(simd, c));
    }
    let (sn, cs) = (sp * x, cp);
    // Quadrant q: (cos, sin) = (C, S), (-S, C), (-C, -S), (S, -C).
    let odd: S::mask32s = S::mask32s::from_vector(((p << 1u32).bitcast::<S::i32s>()) >> 31u32);
    let sign2 = (p & 0x8000_0000u32).bitcast::<S::f32s>();
    let sign1 = ((p ^ (p << 1u32)) & 0x8000_0000u32).bitcast::<S::f32s>();
    let first = odd.select(sn, cs);
    let second = odd.select(cs, sn);
    (
        (first.bitcast::<S::u32s>() ^ sign1.bitcast::<S::u32s>()).bitcast::<S::f32s>(),
        (second.bitcast::<S::u32s>() ^ sign2.bitcast::<S::u32s>()).bitcast::<S::f32s>(),
    )
}

/// One vector pair of rotated samples: `h` samples from the interleaved floats `a ++ b`.
#[inline(always)]
fn rotate_vector<S: SimdToken>(
    simd: S,
    a: &[f32],
    b: &[f32],
    base: u32,
    ramp: S::u32s,
) -> (S::f32s, S::f32s) {
    let (xr, xi) = S::f32s::from_slice(simd, a).deinterleave(S::f32s::from_slice(simd, b));
    let p = S::u32s::splat(simd, base) + ramp;
    let (c, s) = cos_sin_poly(simd, p);
    let yr = xr.mul_sub(c, xi * s);
    let yi = xr.mul_add(s, xi * c);
    yr.interleave(yi)
}

/// Rotates all samples with the polynomial oscillator; the last partial vector is processed
/// zero-padded so that every sample goes through the same arithmetic.
#[inline(always)]
fn rotate_f32<S: SimdToken>(simd: S, phase: u32, inc: u32, input: &[f32], out: &mut [f32]) {
    let h = S::f32s::LEN;
    let ramp = S::u32s::from_fn(simd, |l| (l as u32).wrapping_mul(inc));
    let step = (h as u32).wrapping_mul(inc);
    let mut base = phase;
    let mut in_chunks = input.chunks_exact(2 * h);
    let mut out_chunks = out.chunks_exact_mut(2 * h);
    for (i, o) in (&mut in_chunks).zip(&mut out_chunks) {
        let (a, b) = i.split_at(h);
        let (o0, o1) = rotate_vector::<S>(simd, a, b, base, ramp);
        let (d0, d1) = o.split_at_mut(h);
        o0.store_slice(d0);
        o1.store_slice(d1);
        base = base.wrapping_add(step);
    }
    let (tail_in, tail_out) = (in_chunks.remainder(), out_chunks.into_remainder());
    if !tail_in.is_empty() {
        let mut tmp_in = [0f32; 32];
        let mut tmp_out = [0f32; 32];
        if let Some(dst) = tmp_in.get_mut(..tail_in.len()) {
            dst.copy_from_slice(tail_in);
        }
        if let (Some(i), Some(o)) = (tmp_in.get(..2 * h), tmp_out.get_mut(..2 * h)) {
            let (a, b) = i.split_at(h);
            let (o0, o1) = rotate_vector::<S>(simd, a, b, base, ramp);
            let (d0, d1) = o.split_at_mut(h);
            o0.store_slice(d0);
            o1.store_slice(d1);
        }
        for (d, s) in tail_out.iter_mut().zip(&tmp_out) {
            *d = *s;
        }
    }
}

/// Rotates whole vectors of `CQ15` samples (`h` samples per `i16` vector); returns the number
/// of samples done.
#[inline(always)]
fn rotate_q15<S: SimdToken>(
    simd: S,
    phase: u32,
    inc: u32,
    input: &[i16],
    out: &mut [i16],
) -> usize {
    let h = S::i32s::LEN;
    let ramp = S::u32s::from_fn(simd, |l| (l as u32).wrapping_mul(inc));
    let step = (h as u32).wrapping_mul(inc);
    let mut base = phase;
    let mut done = 0;
    for (i, o) in input.chunks_exact(2 * h).zip(out.chunks_exact_mut(2 * h)) {
        // One 32-bit lane per sample: the low half is I, the high half is Q.
        let x: S::i32s = S::i16s::from_slice(simd, i).bitcast();
        let xr = (x << 16u32) >> 16u32;
        let xi = x >> 16u32;
        let p = S::u32s::splat(simd, base) + ramp;
        let (c, s) = cos_sin_q15(simd, p);
        let round = S::i32s::splat(simd, 1 << 14);
        let yr = (xr * c - xi * s + round) >> 15u32;
        let yi = (xr * s + xi * c + round) >> 15u32;
        let lo = S::i32s::splat(simd, -32768);
        let hi = S::i32s::splat(simd, 32767);
        let yr = yr.max(lo).min(hi);
        let yi = yi.max(lo).min(hi);
        let packed = (yr & 0xFFFFi32) | (yi << 16u32);
        packed.bitcast::<S::i16s>().store_slice(o);
        base = base.wrapping_add(step);
        done += h;
    }
    done
}

impl Rotate<C32> for Simd {
    fn rotate(&self, nco: &mut Nco, input: &[C32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (phase, inc) = (nco.phase(), nco.increment());
        let x: &[f32] = bytemuck::cast_slice(input);
        let y: &mut [f32] = bytemuck::cast_slice_mut(output);
        dispatch!(simd => rotate_f32(simd, phase, inc, x, y));
        let skip = (input.len() as u32).wrapping_mul(inc);
        nco.set_phase(phase.wrapping_add(skip));
        Ok(())
    }
}

impl Rotate<CQ15> for Simd {
    fn rotate(&self, nco: &mut Nco, input: &[CQ15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (phase, inc) = (nco.phase(), nco.increment());
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        let done = dispatch!(simd => rotate_q15(simd, phase, inc, x, y));
        finish(nco, done, input, output)
    }
}

/// Runs the fallback over the samples after the last whole vector and advances the oscillator.
fn finish<T>(nco: &mut Nco, done: usize, input: &[T], output: &mut [T]) -> Result<()>
where
    Fallback: Rotate<T>,
    T: sdrk_core::ComplexSample,
{
    let skip = (done as u32).wrapping_mul(nco.increment());
    nco.set_phase(nco.phase().wrapping_add(skip));
    match (input.get(done..), output.get_mut(done..)) {
        (Some(x), Some(y)) => Fallback.rotate(nco, x, y),
        _ => Err(crate::util::internal("rotate tail")),
    }
}
