//! The entries of the basic arithmetic kernels (milestone M30): `add`, `add_real`, `mul`,
//! `mul_real`, `interleave` and `deinterleave` for the complex types (`C32`, `CQ15`) and the
//! real types (`f32`, `Q15`).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `add` | facade | `volk_32fc_x2_add_32fc` | our fallback |
//! | `add_cq15` | facade | `volk_16i_x2_add_saturated_16i` on the `2 n` values | our fallback |
//! | `add_real` | facade | `volk_32f_x2_add_32f` | our fallback |
//! | `add_real_q15` | facade | `volk_16i_x2_add_saturated_16i` | our fallback |
//! | `mul` | facade | `volk_32fc_x2_multiply_32fc` | our fallback |
//! | `mul_real` | facade | `volk_32f_x2_multiply_32f` | our fallback |
//! | `mul_cq15`, `mul_real_q15` | facade | our fallback (efficiency verdict) | |
//! | `interleave` | facade | `volk_32f_x2_interleave_32fc` | our fallback |
//! | `deinterleave`, `deinterleave_cq15` | facade | `volk_32fc_deinterleave_32f_x2`, `volk_16ic_deinterleave_16i_x2` | our fallback |
//! | `interleave_cq15` | facade | our fallback (efficiency verdict) | |
//! | `add_fallback` ... `mul_real_q15_fallback` | our fallback | CMSIS-DSP generic C | |
//!
//! VOLK's `volk_16ic_x2_multiply_16ic` is not the comparison of `mul_cq15`: it multiplies the
//! 16-bit integers (the low halves of the products wrap, there is no shift by 15), a different
//! function. VOLK has no 16-bit real multiplication and no 16-bit interleave.
//!
//! The planar inputs of `interleave` and the planar outputs of `deinterleave` are separate
//! buffers prepared once. `deinterleave` writes both outputs (the common case of the
//! comparison libraries); `interleave` takes both inputs.

use std::hint::black_box;

use cref_sys::cmsis_dsp::{self, add_f32, add_q15, cmplx_mult_cmplx_f32, cmplx_mult_cmplx_q15};
use sdrk::kernels::math::{Add, AddReal, Interleave, Mul, MulReal};
use sdrk::{C32, CQ15, ComplexSample, Fallback, Q15, Sdrk};
use sdrk_core::RealSample;
use volk_sys::{lv_16sc_t, lv_32fc_t};

use super::{
    Aligned, BiquadEntry, Impl, Kernel, Layout, Pcg32, c32s, cq15s, describe_cmsis,
    dot_scale_supports, noise_c32, snr_db,
};

/// The operation of an arithmetic entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    /// `a + b` (saturating for `Q15`).
    Add,
    /// `a * b` (complex product for the complex types, `q15_mul` for `Q15`).
    Mul,
}

fn noise_q15(rng: &mut Pcg32, n: usize, skew: usize) -> Aligned<Q15> {
    let mut v = Aligned::<Q15>::skewed(n, skew);
    for s in v.iter_mut() {
        *s = Q15((rng.next_u32() >> 16) as i16);
    }
    v
}

fn noise_cq15_full(rng: &mut Pcg32, n: usize, skew: usize) -> Aligned<CQ15> {
    let mut v = Aligned::<CQ15>::skewed(n, skew);
    for s in v.iter_mut() {
        s.re = Q15((rng.next_u32() >> 16) as i16);
        s.im = Q15((rng.next_u32() >> 16) as i16);
    }
    v
}

fn noise_f32(rng: &mut Pcg32, n: usize, amp: f32, skew: usize) -> Aligned<f32> {
    let mut v = Aligned::<f32>::skewed(n, skew);
    for s in v.iter_mut() {
        *s = rng.unit() * amp;
    }
    v
}

fn sat(v: f64) -> f64 {
    v.clamp(-32768.0, 32767.0)
}

fn real_call<T: RealSample, B: AddReal<T> + MulReal<T>>(
    b: &B,
    op: Op,
    x: &[T],
    y: &[T],
    out: &mut [T],
) {
    let _ = black_box(match op {
        Op::Add => b.add_real(black_box(x), black_box(y), out),
        Op::Mul => b.mul_real(black_box(x), black_box(y), out),
    });
}

fn complex_call<T: ComplexSample, B: Add<T> + Mul<T>>(
    b: &B,
    op: Op,
    x: &[T],
    y: &[T],
    out: &mut [T],
) {
    let _ = black_box(match op {
        Op::Add => b.add(black_box(x), black_box(y), out),
        Op::Mul => b.mul(black_box(x), black_box(y), out),
    });
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::Add => "add",
        Op::Mul => "mul",
    }
}

// ---------------------------------------------------------------------------------------------
// f32

/// `add_real` / `mul_real` of `n` floats.
pub(crate) struct RealF32K {
    op: Op,
    a: Aligned<f32>,
    b: Aligned<f32>,
    out: Aligned<f32>,
    entry: BiquadEntry,
}

impl RealF32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry, op: Op) -> Self {
        let skew = layout.skew(4);
        RealF32K {
            op,
            a: noise_f32(rng, layout.n, 0.9, skew),
            b: noise_f32(rng, layout.n, 0.9, skew),
            out: Aligned::skewed(layout.n, skew),
            entry,
        }
    }
}

impl Kernel for RealF32K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.a.len() as u32;
        for _ in 0..k {
            match which {
                Impl::Ours => real_call(&Sdrk, self.op, &self.a, &self.b, &mut self.out),
                Impl::Fallback => real_call(&Fallback, self.op, &self.a, &self.b, &mut self.out),
                // SAFETY: every buffer holds `n` floats and the output aliases no input.
                Impl::Volk => unsafe {
                    let (o, a, b) = (self.out.as_mut_ptr(), self.a.as_ptr(), self.b.as_ptr());
                    match self.op {
                        Op::Add => volk_sys::volk_32f_x2_add_32f(o, a, b, n),
                        Op::Mul => volk_sys::volk_32f_x2_multiply_32f(o, a, b, n),
                    }
                },
                Impl::Cmsis => match self.op {
                    Op::Add => add_f32(&self.a, &self.b, &mut self.out),
                    Op::Mul => cmsis_dsp::mult_f32(&self.a, &self.b, &mut self.out),
                },
                other => unreachable!("{}_real has no {other:?}", op_name(self.op)),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .a
            .iter()
            .zip(self.b.iter())
            .map(|(&a, &b)| match self.op {
                Op::Add => f64::from(a) + f64::from(b),
                Op::Mul => f64::from(a) * f64::from(b),
            })
            .collect();
        snr_db(&want, &super::f32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::FULL)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.op) {
            (Impl::Volk, Op::Add) => "volk_32f_x2_add_32f",
            (Impl::Volk, Op::Mul) => "volk_32f_x2_multiply_32f",
            (Impl::Cmsis, Op::Add) => describe_cmsis("arm_add_f32"),
            (Impl::Cmsis, Op::Mul) => describe_cmsis("arm_mult_f32"),
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        match (self.entry, self.op) {
            (BiquadEntry::Simd, Op::Add) if volk_sys::FULL => &["volk_32f_x2_add_32f"],
            (BiquadEntry::Simd, Op::Mul) if volk_sys::FULL => &["volk_32f_x2_multiply_32f"],
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Q15

/// `add_real` / `mul_real` of `n` `Q15` values over the whole range of `i16`.
pub(crate) struct RealQ15K {
    op: Op,
    a: Aligned<Q15>,
    b: Aligned<Q15>,
    out: Aligned<Q15>,
    entry: BiquadEntry,
}

impl RealQ15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry, op: Op) -> Self {
        let skew = layout.skew(2);
        RealQ15K {
            op,
            a: noise_q15(rng, layout.n, skew),
            b: noise_q15(rng, layout.n, skew),
            out: Aligned::skewed(layout.n, skew),
            entry,
        }
    }
}

fn q15_raw(x: &[Q15]) -> &[i16] {
    // SAFETY: `Q15` is `repr(transparent)` over `i16`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

fn q15_raw_mut(x: &mut [Q15]) -> &mut [i16] {
    // SAFETY: `Q15` is `repr(transparent)` over `i16`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

impl Kernel for RealQ15K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.a.len() as u32;
        for _ in 0..k {
            match which {
                Impl::Ours => real_call(&Sdrk, self.op, &self.a, &self.b, &mut self.out),
                Impl::Fallback => real_call(&Fallback, self.op, &self.a, &self.b, &mut self.out),
                // SAFETY: every buffer holds `n` values and the output aliases no input.
                Impl::Volk if self.op == Op::Add => unsafe {
                    volk_sys::volk_16i_x2_add_saturated_16i(
                        self.out.as_mut_ptr().cast(),
                        self.a.as_ptr().cast(),
                        self.b.as_ptr().cast(),
                        n,
                    );
                },
                Impl::Cmsis => match self.op {
                    Op::Add => add_q15(
                        q15_raw(&self.a),
                        q15_raw(&self.b),
                        q15_raw_mut(&mut self.out),
                    ),
                    Op::Mul => {
                        cmsis_dsp::mult_q15(
                            q15_raw(&self.a),
                            q15_raw(&self.b),
                            q15_raw_mut(&mut self.out),
                        );
                    }
                },
                other => unreachable!("{}_real_q15 has no {other:?}", op_name(self.op)),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .a
            .iter()
            .zip(self.b.iter())
            .map(|(a, b)| {
                let (a, b) = (f64::from(a.0), f64::from(b.0));
                match self.op {
                    Op::Add => sat(a + b),
                    Op::Mul => sat(a * b / 32768.0),
                }
            })
            .collect();
        let got: Vec<f64> = self.out.iter().map(|v| f64::from(v.0)).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::FULL && self.op == Op::Add)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.op) {
            (Impl::Volk, _) => "volk_16i_x2_add_saturated_16i",
            (Impl::Cmsis, Op::Add) => describe_cmsis("arm_add_q15"),
            (Impl::Cmsis, Op::Mul) => describe_cmsis("arm_mult_q15 (truncating)"),
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if self.entry == BiquadEntry::Simd && self.op == Op::Add && volk_sys::FULL {
            &["volk_16i_x2_add_saturated_16i"]
        } else {
            &[]
        }
    }
}

// ---------------------------------------------------------------------------------------------
// C32

/// `add` / `mul` of `n` `C32` samples.
pub(crate) struct ComplexC32K {
    op: Op,
    a: Aligned<C32>,
    b: Aligned<C32>,
    out: Aligned<C32>,
    entry: BiquadEntry,
}

impl ComplexC32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry, op: Op) -> Self {
        let skew = layout.skew(8);
        ComplexC32K {
            op,
            a: noise_c32(rng, layout.n, 0.9, skew),
            b: noise_c32(rng, layout.n, 0.9, skew),
            out: Aligned::skewed(layout.n, skew),
            entry,
        }
    }
}

fn pairs(x: &[C32]) -> &[[f32; 2]] {
    // SAFETY: `C32` is two `f32`, real part first (`repr(C)`).
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

fn pairs_mut(x: &mut [C32]) -> &mut [[f32; 2]] {
    // SAFETY: as in `pairs`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

fn floats(x: &[C32]) -> &[f32] {
    // SAFETY: `C32` is two `f32`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), 2 * x.len()) }
}

fn floats_mut(x: &mut [C32]) -> &mut [f32] {
    // SAFETY: `C32` is two `f32`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), 2 * x.len()) }
}

impl Kernel for ComplexC32K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.a.len() as u32;
        for _ in 0..k {
            match which {
                Impl::Ours => complex_call(&Sdrk, self.op, &self.a, &self.b, &mut self.out),
                Impl::Fallback => {
                    complex_call(&Fallback, self.op, &self.a, &self.b, &mut self.out);
                }
                // SAFETY: every buffer holds `n` complex samples and the output aliases no input.
                Impl::Volk => unsafe {
                    let (o, a, b) = (
                        self.out.as_mut_ptr().cast::<lv_32fc_t>(),
                        self.a.as_ptr().cast::<lv_32fc_t>(),
                        self.b.as_ptr().cast::<lv_32fc_t>(),
                    );
                    match self.op {
                        Op::Add => volk_sys::volk_32fc_x2_add_32fc(o, a, b, n),
                        Op::Mul => volk_sys::volk_32fc_x2_multiply_32fc(o, a, b, n),
                    }
                },
                Impl::Cmsis => match self.op {
                    Op::Add => add_f32(floats(&self.a), floats(&self.b), floats_mut(&mut self.out)),
                    Op::Mul => {
                        cmplx_mult_cmplx_f32(
                            pairs(&self.a),
                            pairs(&self.b),
                            pairs_mut(&mut self.out),
                        );
                    }
                },
                other => unreachable!("{} has no {other:?}", op_name(self.op)),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = Vec::with_capacity(2 * self.a.len());
        for (a, b) in self.a.iter().zip(self.b.iter()) {
            let (ar, ai) = (f64::from(a.re), f64::from(a.im));
            let (br, bi) = (f64::from(b.re), f64::from(b.im));
            match self.op {
                Op::Add => want.extend([ar + br, ai + bi]),
                Op::Mul => want.extend([ar * br - ai * bi, ar * bi + ai * br]),
            }
        }
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::FULL)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.op) {
            (Impl::Volk, Op::Add) => "volk_32fc_x2_add_32fc",
            (Impl::Volk, Op::Mul) => "volk_32fc_x2_multiply_32fc",
            (Impl::Cmsis, Op::Add) => describe_cmsis("arm_add_f32 on the 2 n floats"),
            (Impl::Cmsis, Op::Mul) => describe_cmsis("arm_cmplx_mult_cmplx_f32"),
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        match (self.entry, self.op) {
            (BiquadEntry::Simd, Op::Add) if volk_sys::FULL => &["volk_32fc_x2_add_32fc"],
            (BiquadEntry::Simd, Op::Mul) if volk_sys::FULL => &["volk_32fc_x2_multiply_32fc"],
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------------------------
// CQ15

/// `add` / `mul` of `n` `CQ15` samples over the whole range of `i16`.
pub(crate) struct ComplexCq15K {
    op: Op,
    a: Aligned<CQ15>,
    b: Aligned<CQ15>,
    out: Aligned<CQ15>,
    entry: BiquadEntry,
}

impl ComplexCq15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry, op: Op) -> Self {
        let skew = layout.skew(4);
        ComplexCq15K {
            op,
            a: noise_cq15_full(rng, layout.n, skew),
            b: noise_cq15_full(rng, layout.n, skew),
            out: Aligned::skewed(layout.n, skew),
            entry,
        }
    }
}

fn pairs_q15(x: &[CQ15]) -> &[[i16; 2]] {
    // SAFETY: `CQ15` is two `i16`, real part first.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

fn pairs_q15_mut(x: &mut [CQ15]) -> &mut [[i16; 2]] {
    // SAFETY: as in `pairs_q15`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

fn shorts(x: &[CQ15]) -> &[i16] {
    // SAFETY: `CQ15` is two `i16`.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), 2 * x.len()) }
}

fn shorts_mut(x: &mut [CQ15]) -> &mut [i16] {
    // SAFETY: `CQ15` is two `i16`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), 2 * x.len()) }
}

impl Kernel for ComplexCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.a.len() as u32;
        for _ in 0..k {
            match which {
                Impl::Ours => complex_call(&Sdrk, self.op, &self.a, &self.b, &mut self.out),
                Impl::Fallback => {
                    complex_call(&Fallback, self.op, &self.a, &self.b, &mut self.out);
                }
                // SAFETY: every buffer holds `2 n` values and the output aliases no input.
                Impl::Volk if self.op == Op::Add => unsafe {
                    volk_sys::volk_16i_x2_add_saturated_16i(
                        self.out.as_mut_ptr().cast(),
                        self.a.as_ptr().cast(),
                        self.b.as_ptr().cast(),
                        2 * n,
                    );
                },
                Impl::Cmsis => match self.op {
                    Op::Add => add_q15(shorts(&self.a), shorts(&self.b), shorts_mut(&mut self.out)),
                    Op::Mul => {
                        cmplx_mult_cmplx_q15(
                            pairs_q15(&self.a),
                            pairs_q15(&self.b),
                            pairs_q15_mut(&mut self.out),
                        );
                    }
                },
                other => unreachable!("{}_cq15 has no {other:?}", op_name(self.op)),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = Vec::with_capacity(2 * self.a.len());
        for (a, b) in self.a.iter().zip(self.b.iter()) {
            let (ar, ai) = (f64::from(a.re.0), f64::from(a.im.0));
            let (br, bi) = (f64::from(b.re.0), f64::from(b.im.0));
            match self.op {
                Op::Add => want.extend([sat(ar + br), sat(ai + bi)]),
                Op::Mul => want.extend([
                    sat((ar * br - ai * bi) / 32768.0),
                    sat((ar * bi + ai * br) / 32768.0),
                ]),
            }
        }
        // `arm_cmplx_mult_cmplx_q15` returns Q3.13: scaled by 4 for the accuracy.
        let scale = if which == Impl::Cmsis && self.op == Op::Mul {
            4.0
        } else {
            1.0
        };
        snr_db(&want, &cq15s(&self.out, scale))
    }

    fn supports(&self, which: Impl) -> bool {
        dot_scale_supports(self.entry, which, volk_sys::FULL && self.op == Op::Add)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.op) {
            (Impl::Volk, _) => "volk_16i_x2_add_saturated_16i on the 2 n values",
            (Impl::Cmsis, Op::Add) => describe_cmsis("arm_add_q15 on the 2 n values"),
            (Impl::Cmsis, Op::Mul) => {
                describe_cmsis("arm_cmplx_mult_cmplx_q15 (Q3.13, truncating, scaled by 4)")
            }
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if self.entry == BiquadEntry::Simd && self.op == Op::Add && volk_sys::FULL {
            &["volk_16i_x2_add_saturated_16i"]
        } else {
            &[]
        }
    }
}

// ---------------------------------------------------------------------------------------------
// interleave and deinterleave

/// `interleave` of two planar buffers of `n` floats (`C32`) or `n` `Q15` values (`CQ15`), or
/// `deinterleave` of `n` samples.
pub(crate) struct SplitC32K {
    deinterleave: bool,
    re: Aligned<f32>,
    im: Aligned<f32>,
    packed: Aligned<C32>,
}

impl SplitC32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, deinterleave: bool) -> Self {
        let n = layout.n;
        let packed = noise_c32(rng, n, 0.9, layout.skew(8));
        let mut k = SplitC32K {
            deinterleave,
            re: Aligned::skewed(n, layout.skew(4)),
            im: Aligned::skewed(n, layout.skew(4)),
            packed,
        };
        if !deinterleave {
            for (i, v) in k.packed.iter().enumerate() {
                k.re[i] = v.re;
                k.im[i] = v.im;
            }
            k.packed.iter_mut().for_each(|v| *v = C32::default());
        }
        k
    }
}

#[inline(never)]
fn interleave_call<T: ComplexSample, B: Interleave<T>>(
    b: &B,
    re: &[T::Real],
    im: &[T::Real],
    out: &mut [T],
) {
    let _ = black_box(b.interleave(black_box(re), Some(black_box(im)), out));
}

#[inline(never)]
fn deinterleave_call<T: ComplexSample, B: Interleave<T>>(
    b: &B,
    input: &[T],
    re: &mut [T::Real],
    im: &mut [T::Real],
) {
    let _ = black_box(b.deinterleave(black_box(input), Some(re), Some(im)));
}

impl Kernel for SplitC32K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.re.len() as u32;
        for _ in 0..k {
            match (which, self.deinterleave) {
                (Impl::Ours, false) => interleave_call(&Sdrk, &self.re, &self.im, &mut self.packed),
                (Impl::Fallback, false) => {
                    interleave_call(&Fallback, &self.re, &self.im, &mut self.packed);
                }
                (Impl::Ours, true) => {
                    deinterleave_call(&Sdrk, &self.packed, &mut self.re, &mut self.im)
                }
                (Impl::Fallback, true) => {
                    deinterleave_call(&Fallback, &self.packed, &mut self.re, &mut self.im);
                }
                // SAFETY: every buffer holds `n` floats (`n` complex samples for the packed one).
                (Impl::Volk, false) => unsafe {
                    volk_sys::volk_32f_x2_interleave_32fc(
                        self.packed.as_mut_ptr().cast::<lv_32fc_t>(),
                        self.re.as_ptr(),
                        self.im.as_ptr(),
                        n,
                    );
                },
                // SAFETY: as above.
                (Impl::Volk, true) => unsafe {
                    volk_sys::volk_32fc_deinterleave_32f_x2(
                        self.re.as_mut_ptr(),
                        self.im.as_mut_ptr(),
                        self.packed.as_ptr().cast::<lv_32fc_t>(),
                        n,
                    );
                },
                (other, _) => unreachable!("interleave has no {other:?}"),
            }
            black_box(&self.packed);
            black_box(&self.re);
            black_box(&self.im);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        if self.deinterleave {
            self.re.iter_mut().for_each(|v| *v = 0.0);
            self.im.iter_mut().for_each(|v| *v = 0.0);
        } else {
            self.packed.iter_mut().for_each(|v| *v = C32::default());
        }
        self.run(which, 1);
        let mut want = Vec::with_capacity(2 * self.re.len());
        let mut got = Vec::with_capacity(2 * self.re.len());
        for (i, v) in self.packed.iter().enumerate() {
            let (r, m) = (f64::from(self.re[i]), f64::from(self.im[i]));
            want.extend([r, m]);
            got.extend([f64::from(v.re), f64::from(v.im)]);
        }
        if self.deinterleave {
            // The packed input is the golden; the planar outputs are the result.
            std::mem::swap(&mut want, &mut got);
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback) || (which == Impl::Volk && volk_sys::FULL)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.deinterleave) {
            (Impl::Volk, false) => "volk_32f_x2_interleave_32fc",
            (Impl::Volk, true) => "volk_32fc_deinterleave_32f_x2",
            (Impl::Fallback, _) => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        match (volk_sys::FULL, self.deinterleave) {
            (true, false) => &["volk_32f_x2_interleave_32fc"],
            (true, true) => &["volk_32fc_deinterleave_32f_x2"],
            _ => &[],
        }
    }
}

/// [`SplitC32K`] for `CQ15` and `Q15`.
pub(crate) struct SplitCq15K {
    deinterleave: bool,
    re: Aligned<Q15>,
    im: Aligned<Q15>,
    packed: Aligned<CQ15>,
}

impl SplitCq15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, deinterleave: bool) -> Self {
        let n = layout.n;
        let packed = noise_cq15_full(rng, n, layout.skew(4));
        let mut k = SplitCq15K {
            deinterleave,
            re: Aligned::skewed(n, layout.skew(2)),
            im: Aligned::skewed(n, layout.skew(2)),
            packed,
        };
        if !deinterleave {
            for (i, v) in k.packed.iter().enumerate() {
                k.re[i] = v.re;
                k.im[i] = v.im;
            }
            k.packed.iter_mut().for_each(|v| *v = CQ15::default());
        }
        k
    }
}

impl Kernel for SplitCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        let n = self.re.len() as u32;
        for _ in 0..k {
            match (which, self.deinterleave) {
                (Impl::Ours, false) => interleave_call(&Sdrk, &self.re, &self.im, &mut self.packed),
                (Impl::Fallback, false) => {
                    interleave_call(&Fallback, &self.re, &self.im, &mut self.packed);
                }
                (Impl::Ours, true) => {
                    deinterleave_call(&Sdrk, &self.packed, &mut self.re, &mut self.im)
                }
                (Impl::Fallback, true) => {
                    deinterleave_call(&Fallback, &self.packed, &mut self.re, &mut self.im);
                }
                // SAFETY: every buffer holds `n` values (`n` complex samples for the packed one).
                (Impl::Volk, true) => unsafe {
                    volk_sys::volk_16ic_deinterleave_16i_x2(
                        self.re.as_mut_ptr().cast(),
                        self.im.as_mut_ptr().cast(),
                        self.packed.as_ptr().cast::<lv_16sc_t>(),
                        n,
                    );
                },
                (other, _) => unreachable!("interleave_cq15 has no {other:?}"),
            }
            black_box(&self.packed);
            black_box(&self.re);
            black_box(&self.im);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        if self.deinterleave {
            self.re.iter_mut().for_each(|v| *v = Q15(0));
            self.im.iter_mut().for_each(|v| *v = Q15(0));
        } else {
            self.packed.iter_mut().for_each(|v| *v = CQ15::default());
        }
        self.run(which, 1);
        let mut want = Vec::with_capacity(2 * self.re.len());
        let mut got = Vec::with_capacity(2 * self.re.len());
        for (i, v) in self.packed.iter().enumerate() {
            want.extend([f64::from(self.re[i].0), f64::from(self.im[i].0)]);
            got.extend([f64::from(v.re.0), f64::from(v.im.0)]);
        }
        if self.deinterleave {
            std::mem::swap(&mut want, &mut got);
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
            || (which == Impl::Volk && volk_sys::FULL && self.deinterleave)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Volk => "volk_16ic_deinterleave_16i_x2",
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }

    fn volk_functions(&self) -> &'static [&'static str] {
        if volk_sys::FULL && self.deinterleave {
            &["volk_16ic_deinterleave_16i_x2"]
        } else {
            &[]
        }
    }
}

// ---------------------------------------------------------------------------------------------
// oscillate

use sdrk::kernels::mix::{Nco, Oscillate};

/// The workload of `oscillate`: 19 kHz at 192 kHz, harmonic 1, from phase 0 on every run.
const OSC_INC: u32 = ((19_000u64 << 32) / 192_000) as u32;

/// The oscillator as a CMSIS-DSP user writes it (CMSIS-DSP has no NCO): the phase accumulator
/// of the loop and `arm_sin_cos_f32` of the angle in degrees per sample.
#[inline(never)]
fn cmsis_oscillate_c32(out: &mut [C32]) {
    let mut phase = 0u32;
    for o in out.iter_mut() {
        // The signed phase is the angle in [-180, 180) degrees.
        let theta = (phase as i32 as f32) * (180.0 / 2_147_483_648.0);
        let (mut s, mut c) = (0.0f32, 0.0f32);
        // SAFETY: plain scalar function of CMSIS-DSP writing two floats; the callers check
        // `cmsis_dsp::AVAILABLE`.
        unsafe { cmsis_dsp::ffi::arm_sin_cos_f32(theta, &raw mut s, &raw mut c) };
        *o = C32::new(c, s);
        phase = phase.wrapping_add(OSC_INC);
    }
}

/// The `CQ15` oscillator with `arm_cos_q15` and `arm_sin_q15` of the angle `phase >> 17` per
/// sample (512-entry table, linear interpolation).
#[inline(never)]
fn cmsis_oscillate_cq15(out: &mut [CQ15]) {
    let mut phase = 0u32;
    for o in out.iter_mut() {
        let x = (phase >> 17) as i16;
        // SAFETY: plain scalar functions of CMSIS-DSP; the callers check `cmsis_dsp::AVAILABLE`.
        let (c, s) = unsafe {
            (
                cmsis_dsp::ffi::arm_cos_q15(x),
                cmsis_dsp::ffi::arm_sin_q15(x),
            )
        };
        *o = CQ15 {
            re: Q15(c),
            im: Q15(s),
        };
        phase = phase.wrapping_add(OSC_INC);
    }
}

#[inline(never)]
fn oscillate_call<T: ComplexSample, B: Oscillate<T>>(b: &B, out: &mut [T]) {
    let mut nco = Nco::from_increment(OSC_INC);
    let _ = black_box(b.oscillate(&mut nco, black_box(1), out));
    black_box(nco);
}

/// `oscillate` of `n` `C32` samples.
pub(crate) struct OscC32K {
    out: Aligned<C32>,
    entry: BiquadEntry,
}

impl OscC32K {
    pub(crate) fn new(layout: Layout, entry: BiquadEntry) -> Self {
        OscC32K {
            out: Aligned::skewed(layout.n, layout.skew(8)),
            entry,
        }
    }
}

impl Kernel for OscC32K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => oscillate_call(&Sdrk, &mut self.out),
                Impl::Fallback => oscillate_call(&Fallback, &mut self.out),
                Impl::Cmsis => cmsis_oscillate_c32(&mut self.out),
                other => unreachable!("oscillate has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = Vec::with_capacity(2 * self.out.len());
        let mut phase = 0u32;
        for _ in 0..self.out.len() {
            let a = f64::from(phase) * std::f64::consts::TAU / 4_294_967_296.0;
            want.extend([a.cos(), a.sin()]);
            phase = phase.wrapping_add(OSC_INC);
        }
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (_, Impl::Fallback) | (BiquadEntry::Simd, Impl::Ours) => true,
            (BiquadEntry::Fallback, Impl::Cmsis) => cmsis_dsp::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(
                "arm_sin_cos_f32 of the angle in degrees per sample (360-entry table, linear interpolation)",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

/// `oscillate` of `n` `CQ15` samples.
pub(crate) struct OscCq15K {
    out: Aligned<CQ15>,
    entry: BiquadEntry,
}

impl OscCq15K {
    pub(crate) fn new(layout: Layout, entry: BiquadEntry) -> Self {
        OscCq15K {
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for OscCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => oscillate_call(&Sdrk, &mut self.out),
                Impl::Fallback => oscillate_call(&Fallback, &mut self.out),
                Impl::Cmsis => cmsis_oscillate_cq15(&mut self.out),
                other => unreachable!("oscillate_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = Vec::with_capacity(2 * self.out.len());
        let mut phase = 0u32;
        for _ in 0..self.out.len() {
            let a = f64::from(phase) * std::f64::consts::TAU / 4_294_967_296.0;
            want.extend([sat(a.cos() * 32768.0), sat(a.sin() * 32768.0)]);
            phase = phase.wrapping_add(OSC_INC);
        }
        snr_db(&want, &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        match (self.entry, which) {
            (_, Impl::Fallback) | (BiquadEntry::Simd, Impl::Ours) => true,
            (BiquadEntry::Fallback, Impl::Cmsis) => cmsis_dsp::AVAILABLE,
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(
                "arm_cos_q15 + arm_sin_q15 per sample (512-entry table, linear interpolation)",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}
