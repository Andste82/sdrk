//! Sample-type specific helpers of the blocks.
//!
//! The blocks are generic over the four sample types. The two traits here collect the few
//! operations that differ between the `f32` and the Q15 family (tap conversion, gains,
//! interpolation); everything else is a kernel call on the facade.

use core::f64::consts::PI;

use sdrk::Sdrk;
use sdrk_core::kernels::dot::{Dot, DotReal};
use sdrk_core::num::{round_shift, sat16};
use sdrk_core::{C32, CQ15, ComplexSample, Q15, RealSample, Sample};

use crate::design::q15_tap;
use crate::error::Result;

/// Operations of the tuned (polyphase) resampler for one sample type.
pub trait TunedElem: Sample {
    /// Type of the filter taps (`f32` or `Q15`).
    type Tap: Sample;

    /// Converts a designed tap (Q15: with the range check of `design::to_q15`).
    fn tap(v: f64) -> Result<Self::Tap>;

    /// `Σ x[i] taps[i]` through the facade kernel `dot` or `dot_real`.
    fn dot(x: &[Self], taps: &[Self::Tap]) -> Result<Self>;

    /// `a (1 - mu / 65536) + b mu / 65536` for `mu` in `0..65536`: `f32` types compute in `f32`,
    /// Q15 types `round_shift(a (65536 - mu) + b mu, 16)` in `i64` with saturation.
    fn lerp(a: Self, b: Self, mu: u32) -> Self;
}

fn lerp_f32(a: f32, b: f32, m: f32) -> f32 {
    a * (1.0 - m) + b * m
}

fn lerp_q15(a: Q15, b: Q15, mu: u32) -> Q15 {
    let mu = i64::from(mu);
    let v = i64::from(a.0) * (65536 - mu) + i64::from(b.0) * mu;
    Q15(sat16(round_shift(v, 16)))
}

impl TunedElem for C32 {
    type Tap = f32;

    fn tap(v: f64) -> Result<f32> {
        Ok(v as f32)
    }

    fn dot(x: &[C32], taps: &[f32]) -> Result<C32> {
        Ok(<Sdrk as Dot<C32>>::dot(&Sdrk, x, taps)?)
    }

    fn lerp(a: C32, b: C32, mu: u32) -> C32 {
        let m = mu as f32 / 65536.0;
        C32::new(lerp_f32(a.re, b.re, m), lerp_f32(a.im, b.im, m))
    }
}

impl TunedElem for CQ15 {
    type Tap = Q15;

    fn tap(v: f64) -> Result<Q15> {
        q15_tap(v)
    }

    fn dot(x: &[CQ15], taps: &[Q15]) -> Result<CQ15> {
        Ok(<Sdrk as Dot<CQ15>>::dot(&Sdrk, x, taps)?)
    }

    fn lerp(a: CQ15, b: CQ15, mu: u32) -> CQ15 {
        CQ15 {
            re: lerp_q15(a.re, b.re, mu),
            im: lerp_q15(a.im, b.im, mu),
        }
    }
}

impl TunedElem for f32 {
    type Tap = f32;

    fn tap(v: f64) -> Result<f32> {
        Ok(v as f32)
    }

    fn dot(x: &[f32], taps: &[f32]) -> Result<f32> {
        Ok(<Sdrk as DotReal<f32>>::dot_real(&Sdrk, x, taps)?)
    }

    fn lerp(a: f32, b: f32, mu: u32) -> f32 {
        lerp_f32(a, b, mu as f32 / 65536.0)
    }
}

impl TunedElem for Q15 {
    type Tap = Q15;

    fn tap(v: f64) -> Result<Q15> {
        q15_tap(v)
    }

    fn dot(x: &[Q15], taps: &[Q15]) -> Result<Q15> {
        Ok(<Sdrk as DotReal<Q15>>::dot_real(&Sdrk, x, taps)?)
    }

    fn lerp(a: Q15, b: Q15, mu: u32) -> Q15 {
        lerp_q15(a, b, mu)
    }
}

/// A real sample type the blocks work with: `f32` or `Q15`.
pub trait BlockReal: RealSample + TunedElem<Tap = Self> {}

impl BlockReal for f32 {}
impl BlockReal for Q15 {}

/// A complex sample type the blocks work with: `C32` or `CQ15`.
pub trait BlockSample: ComplexSample + TunedElem<Tap = <Self as ComplexSample>::Real> {
    /// Largest gain of the `scale` block in dB: `+40` for `C32`, `0` for `CQ15` (Q15 cannot
    /// represent a gain above 1).
    const MAX_GAIN_DB: f64;

    /// Gain of the FM discriminator for the deviation and the rate: `C32` uses
    /// `fs / (2 pi deviation)` (the full deviation reads 1.0); `CQ15` uses the largest gain,
    /// raw 32767 (the output unit is phase / pi).
    fn demod_gain(deviation_hz: f64, rate_hz: f64) -> Self::Real;
}

impl BlockSample for C32 {
    const MAX_GAIN_DB: f64 = 40.0;

    fn demod_gain(deviation_hz: f64, rate_hz: f64) -> f32 {
        (rate_hz / (2.0 * PI * deviation_hz)) as f32
    }
}

impl BlockSample for CQ15 {
    const MAX_GAIN_DB: f64 = 0.0;

    fn demod_gain(_deviation_hz: f64, _rate_hz: f64) -> Q15 {
        Q15(32767)
    }
}
