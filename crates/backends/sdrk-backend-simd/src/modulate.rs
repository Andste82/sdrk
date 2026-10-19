//! `fm_mod`: frequency modulation of a real baseband signal.
//!
//! The phase steps `floor(x * k + 0.5)` are computed exactly like the fallback (in `f64`,
//! wrapped into `u32`), the running phase is a wrapping prefix sum, and the oscillator values
//! come from the same table lookup as `rotate`. Bit-exact with the fallback.

use fearless_simd::{Simd as SimdToken, prelude::*};
use sdrk_core::kernels::modulate::{FmMod, FmModState};
use sdrk_core::{C32, CQ15, Fallback, Q15, Result};

use crate::Simd;
use crate::level::dispatch;
use crate::table::{cos_sin_f32, cos_sin_q15};
use crate::util::check_len;

/// `1.5 * 2^52`: adding it to an integer-valued `f64` of magnitude below `2^51` leaves the
/// two's-complement integer in the low mantissa bits.
const MAGIC: f64 = 6_755_399_441_055_744.0;
/// Largest magnitude of a step that the magic-number trick handles.
const LIMIT: f64 = 2_251_799_813_685_248.0;

/// The wrapped phase steps `(floor(x * k + 0.5) as i64) as u32` of `h` f32 values, or `None`
/// when a step is out of the exactly handled range or NaN (the caller then takes the scalar
/// path).
#[inline(always)]
fn steps<S: SimdToken>(simd: S, x: S::f32s, k: f64) -> Option<S::u32s> {
    let (lo, hi) = x.widen();
    let kk = S::f64s::splat(simd, k);
    let t_lo = (lo * kk + 0.5).floor();
    let t_hi = (hi * kk + 0.5).floor();
    let limit = S::f64s::splat(simd, LIMIT);
    let ok = t_lo.abs().simd_lt(limit) & t_hi.abs().simd_lt(limit);
    if !ok.all_true() {
        return None;
    }
    let magic = S::f64s::splat(simd, MAGIC);
    let a: S::u32s = (t_lo + magic).bitcast();
    let b: S::u32s = (t_hi + magic).bitcast();
    Some(a.unzip_low(b))
}

/// Inclusive wrapping prefix sum over the lanes.
#[inline(always)]
fn prefix_sum<S: SimdToken>(v: S::u32s) -> S::u32s {
    let h = S::u32s::LEN;
    let mut s = v;
    if h > 1 {
        s = s + s.shift_elements_right::<1>(0);
    }
    if h > 2 {
        s = s + s.shift_elements_right::<2>(0);
    }
    if h > 4 {
        s = s + s.shift_elements_right::<4>(0);
    }
    if h > 8 {
        s = s + s.shift_elements_right::<8>(0);
    }
    s
}

/// Modulates whole vectors; returns the number of samples done and the phase after them.
#[inline(always)]
fn fm_mod_f32<S: SimdToken>(
    simd: S,
    k: f64,
    phase: u32,
    input: &[f32],
    out: &mut [f32],
) -> (usize, u32) {
    let h = S::f32s::LEN;
    let mut phase = phase;
    let mut done = 0;
    for (i, o) in input.chunks_exact(h).zip(out.chunks_exact_mut(2 * h)) {
        let Some(st) = steps(simd, S::f32s::from_slice(simd, i), k) else {
            break;
        };
        let p = S::u32s::splat(simd, phase) + prefix_sum::<S>(st);
        let (c, s) = cos_sin_f32(simd, p);
        let (o0, o1) = c.interleave(s);
        let (d0, d1) = o.split_at_mut(h);
        o0.store_slice(d0);
        o1.store_slice(d1);
        phase = p.as_slice().last().copied().unwrap_or(phase);
        done += h;
    }
    (done, phase)
}

/// `CQ15` variant: `h` samples per vector of `i32` lanes.
#[inline(always)]
fn fm_mod_q15<S: SimdToken>(
    simd: S,
    k: f64,
    phase: u32,
    input: &[i16],
    out: &mut [i16],
) -> (usize, u32) {
    let h = S::i32s::LEN;
    let mut phase = phase;
    let mut done = 0;
    for (i, o) in input.chunks_exact(h).zip(out.chunks_exact_mut(2 * h)) {
        // `i16` to `f32` is exact; `q / 32768` in `f64` is exact too.
        let x = S::i32s::from_fn(simd, {
            let mut it = i.iter();
            move |_| it.next().map_or(0, |&v| i32::from(v))
        })
        .to_float::<S::f32s>()
            * (1.0 / 32768.0);
        let Some(st) = steps(simd, x, k) else {
            break;
        };
        let p = S::u32s::splat(simd, phase) + prefix_sum::<S>(st);
        let (c, s) = cos_sin_q15(simd, p);
        let packed = (c & 0xFFFFi32) | (s << 16u32);
        packed.bitcast::<S::i16s>().store_slice(o);
        phase = p.as_slice().last().copied().unwrap_or(phase);
        done += h;
    }
    (done, phase)
}

impl FmMod<C32> for Simd {
    fn fm_mod(&self, s: &mut FmModState, input: &[f32], output: &mut [C32]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (k, phase) = s.parts_mut();
        let y: &mut [f32] = bytemuck::cast_slice_mut(output);
        let (done, p) = dispatch!(simd => fm_mod_f32(simd, k, *phase, input, y));
        *phase = p;
        match (input.get(done..), output.get_mut(done..)) {
            (Some(x), Some(y)) => Fallback.fm_mod(s, x, y),
            _ => Err(crate::util::internal("fm_mod tail")),
        }
    }
}

impl FmMod<CQ15> for Simd {
    fn fm_mod(&self, s: &mut FmModState, input: &[Q15], output: &mut [CQ15]) -> Result<()> {
        check_len(input.len(), output.len())?;
        let (k, phase) = s.parts_mut();
        let x: &[i16] = bytemuck::cast_slice(input);
        let y: &mut [i16] = bytemuck::cast_slice_mut(output);
        let (done, p) = dispatch!(simd => fm_mod_q15(simd, k, *phase, x, y));
        *phase = p;
        match (input.get(done..), output.get_mut(done..)) {
            (Some(x), Some(y)) => Fallback.fm_mod(s, x, y),
            _ => Err(crate::util::internal("fm_mod tail")),
        }
    }
}
