//! The entries of the `am_mod` kernel (milestone M35): `out[n] = (c (1 + m x'), 0)` with `x'` the
//! input clamped to `[-1, 1]`, for `f32` to `C32` and `Q15` to `CQ15`.
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `am_mod` | facade | our fallback (efficiency verdict) | |
//! | `am_mod_cq15` | facade | our fallback (efficiency verdict) | the CMSIS-DSP composition |
//! | `am_mod_fallback` | our fallback | liquid-dsp 1.8.3 `ampmodem_modulate_block` (DSB with the carrier, scalar C) | |
//! | `am_mod_cq15_fallback` | our fallback | the CMSIS-DSP composition (generic C) | |
//!
//! No SIMD library has an AM modulator (VOLK, KFR and CMSIS-DSP have none; liquid-dsp's
//! `ampmodem` is scalar C), so the facade entries compare with our fallback. liquid's DSB
//! modulator computes `x mod_index + 1` per sample as a complex value: the carrier amplitude is
//! 1, and it neither clamps the input nor multiplies by a carrier, so it does the per-sample work
//! of ours without the clamp and the carrier product. It runs with `mod_index` = `depth`; its
//! accuracy is that of its output times `carrier` against the `f64` model.
//!
//! CMSIS-DSP has no AM modulator and no real-to-complex copy: the composition is
//! `arm_scale_q15` by `round(c m 32768)` (truncating `>> 15`), `arm_offset_q15` by `c` in place
//! (saturating) and a plain loop that writes the pairs `(re, 0)`, `c (1 + m x) = c + c m x` in
//! another rounding than ours (one truncated product instead of two rounded ones).
//!
//! The workload is the block default (carrier 0.5, depth 0.8) on a 1 kHz tone of 0.6 at 48 kHz
//! plus uniform noise of 0.3, inside `[-0.9, 0.9]` so that liquid's output, which does not
//! clamp, is comparable; the `Q15` input is the same signal rounded to `Q15`.

use std::hint::black_box;

use cref_sys::LiquidAmpmodem;
use cref_sys::cmsis_dsp;
use sdrk::kernels::modulate::AmMod;
use sdrk::{C32, CQ15, ComplexSample, Fallback, Q15, Sdrk};

use super::{
    Aligned, BiquadEntry, Impl, Kernel, Layout, Pcg32, c32s, cq15s, describe_cmsis, snr_db,
};

/// The carrier amplitude of the workload (the block default).
const CARRIER: f32 = 0.5;
/// The modulation index of the workload (the block default).
const DEPTH: f32 = 0.8;

/// The audio of the workload: a 1 kHz tone of 0.6 at 48 kHz plus uniform noise of 0.3.
fn audio(rng: &mut Pcg32, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let tone = 0.6 * (std::f64::consts::TAU * 1000.0 * i as f64 / 48_000.0).sin();
            tone + 0.3 * f64::from(rng.unit())
        })
        .collect()
}

/// The `f64` model of the contract: `carrier (1 + depth clamp(x, -1, 1))`.
fn model(x: f64) -> f64 {
    f64::from(CARRIER) * (1.0 + f64::from(DEPTH) * x.clamp(-1.0, 1.0))
}

#[inline(never)]
fn am_mod_call<T: ComplexSample, B: AmMod<T>>(b: &B, input: &[T::Real], out: &mut [T]) {
    let _ = black_box(b.am_mod(black_box(CARRIER), black_box(DEPTH), black_box(input), out));
}

fn am_supports(entry: BiquadEntry, which: Impl, reference: Impl, has: bool) -> bool {
    match (entry, which) {
        (BiquadEntry::Simd, Impl::Ours) | (_, Impl::Fallback) => true,
        (_, w) if w == reference => has,
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------
// C32

/// `am_mod` of `n` `f32` samples into `C32`; the fallback entry against liquid-dsp's
/// `ampmodem`.
pub(crate) struct AmModC32K {
    input: Aligned<f32>,
    out: Aligned<C32>,
    entry: BiquadEntry,
}

impl AmModC32K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let x = audio(rng, layout.n);
        let mut input = Aligned::<f32>::skewed(layout.n, layout.skew(4));
        for (s, v) in input.iter_mut().zip(&x) {
            *s = *v as f32;
        }
        AmModC32K {
            input,
            out: Aligned::skewed(layout.n, layout.skew(8)),
            entry,
        }
    }

    /// liquid's modulator, created per run so that its cost cancels between the counted runs.
    fn liquid(which: Impl) -> Option<LiquidAmpmodem> {
        (which == Impl::Liquid)
            .then(|| LiquidAmpmodem::new(DEPTH))
            .flatten()
    }
}

/// The `C32` buffer as liquid's `float complex` pairs.
fn pairs_mut(x: &mut [C32]) -> &mut [[f32; 2]] {
    // SAFETY: `C32` is `repr(C)` of two `f32`, the layout of `[f32; 2]`.
    unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

impl Kernel for AmModC32K {
    fn run(&mut self, which: Impl, k: usize) {
        let mut liquid = Self::liquid(which);
        for _ in 0..k {
            match which {
                Impl::Ours => am_mod_call(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => am_mod_call(&Fallback, &self.input, &mut self.out),
                Impl::Liquid => liquid
                    .as_mut()
                    .expect("liquid ampmodem")
                    .modulate_block(black_box(&self.input), pairs_mut(&mut self.out)),
                other => unreachable!("am_mod has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .flat_map(|&x| [model(f64::from(x)), 0.0])
            .collect();
        let mut got = c32s(&self.out);
        if which == Impl::Liquid {
            // liquid's carrier amplitude is 1: its output times `carrier` is the same signal.
            for v in &mut got {
                *v *= f64::from(CARRIER);
            }
        }
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        am_supports(
            self.entry,
            which,
            Impl::Liquid,
            self.entry == BiquadEntry::Fallback && LiquidAmpmodem::new(DEPTH).is_some(),
        )
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Liquid => {
                "liquid-dsp 1.8.3 ampmodem_modulate_block (LIQUID_AMPMODEM_DSB with the carrier: x * mod_index + 1, no clamp, carrier 1; -O3)"
            }
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// CQ15

/// `am_mod` of `n` `Q15` samples into `CQ15`; CMSIS-DSP's composition as information on the
/// facade entry and as the gate of the fallback entry.
pub(crate) struct AmModCq15K {
    input: Aligned<Q15>,
    /// The real part of CMSIS-DSP's composition.
    tmp: Aligned<i16>,
    out: Aligned<CQ15>,
    entry: BiquadEntry,
}

/// The `Q15` value of `v` (round half away from zero, saturated).
fn q15_of(v: f64) -> i16 {
    (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// CMSIS-DSP's scale `round(carrier depth 32768)` of the composition.
fn cmsis_scale() -> i16 {
    q15_of(f64::from(CARRIER) * f64::from(DEPTH))
}

/// CMSIS-DSP's offset `round(carrier 32768)` of the composition.
fn cmsis_offset() -> i16 {
    q15_of(f64::from(CARRIER))
}

/// The CMSIS-DSP composition: `arm_scale_q15` (`c m x`, truncating), `arm_offset_q15` by `c` in
/// place, and the pairs `(re, 0)` in a plain loop, the raw functions without the length checks
/// of the wrappers.
#[inline(never)]
fn cmsis_am_mod_q15(input: &[Q15], tmp: &mut [i16], out: &mut [CQ15], scale: i16, offset: i16) {
    let n = out.len();
    assert!(input.len() == n && tmp.len() == n);
    let len = u32::try_from(n).expect("block");
    // SAFETY: every buffer holds `n` values; `Q15` is `repr(transparent)` over `i16`; the
    // offset runs in place, which CMSIS-DSP's element-wise functions allow.
    unsafe {
        cmsis_dsp::ffi::arm_scale_q15(input.as_ptr().cast(), scale, 0, tmp.as_mut_ptr(), len);
        cmsis_dsp::ffi::arm_offset_q15(tmp.as_ptr(), offset, tmp.as_mut_ptr(), len);
    }
    for (o, &re) in out.iter_mut().zip(tmp.iter()) {
        *o = CQ15 {
            re: Q15(re),
            im: Q15(0),
        };
    }
}

impl AmModCq15K {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, entry: BiquadEntry) -> Self {
        let x = audio(rng, layout.n);
        let mut input = Aligned::<Q15>::skewed(layout.n, layout.skew(2));
        for (s, v) in input.iter_mut().zip(&x) {
            *s = Q15(q15_of(*v));
        }
        AmModCq15K {
            input,
            tmp: Aligned::skewed(layout.n, layout.skew(2)),
            out: Aligned::skewed(layout.n, layout.skew(4)),
            entry,
        }
    }
}

impl Kernel for AmModCq15K {
    fn run(&mut self, which: Impl, k: usize) {
        let (scale, offset) = (cmsis_scale(), cmsis_offset());
        for _ in 0..k {
            match which {
                Impl::Ours => am_mod_call(&Sdrk, &self.input, &mut self.out),
                Impl::Fallback => am_mod_call(&Fallback, &self.input, &mut self.out),
                Impl::Cmsis => cmsis_am_mod_q15(
                    black_box(&self.input),
                    &mut self.tmp,
                    &mut self.out,
                    scale,
                    offset,
                ),
                other => unreachable!("am_mod_cq15 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want: Vec<f64> = self
            .input
            .iter()
            .flat_map(|x| {
                let re = (model(f64::from(x.0) / 32768.0) * 32768.0).clamp(-32768.0, 32767.0);
                [re, 0.0]
            })
            .collect();
        snr_db(&want, &cq15s(&self.out, 1.0))
    }

    fn supports(&self, which: Impl) -> bool {
        am_supports(self.entry, which, Impl::Cmsis, cmsis_dsp::AVAILABLE)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Cmsis => describe_cmsis(
                "arm_scale_q15 by round(c m 32768) (truncating), arm_offset_q15 by c in place and a plain loop to (re, 0) (a composition: CMSIS-DSP has no AM modulator)",
            ),
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}
