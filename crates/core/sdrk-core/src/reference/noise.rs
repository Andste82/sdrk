//! Reference implementation of the `gauss_noise` kernel: the plain loop of the contract, one
//! real value at a time, every draw taken (also when `sigma` counts as 0). The inverse CDF is
//! written here again in `i64` arithmetic, and the outputs are computed from the exact `f64`
//! value `v = zq · 2^-17 · sigma_e` (`as f32`, and `q15_from_f64`), so this checks the integer
//! `Q15` path and the `f32` scale of the fallback independently.

use crate::Reference;
use crate::error::Result;
use crate::kernels::noise::{GaussNoise, GaussNoiseState, Pcg32};
use crate::num::{flush_denormal, q15_from_f64};
use crate::tables::GAUSS_ICDF;
use crate::types::{C32, CQ15, Q15, Sample};

/// `2^-17`.
const TWO_POW_MINUS_17: f64 = 1.0 / 131_072.0;

/// The Q17 normal value of one draw (§2.1 of the contract) in `i64`.
// The names of the contract's formula (milestone M48 §2.1).
#[allow(clippy::many_single_char_names)]
fn zq(r: u32) -> i64 {
    let r = i64::from(r);
    let negative = r >> 31 == 1;
    // x = 2 · (r mod 2^31) + 1, odd, in [1, 2^32).
    let x = 2 * (r & 0x7FFF_FFFF) + 1;
    // Octave: x in [2^l, 2^(l + 1)).
    let mut l = 31;
    while x >> l == 0 {
        l -= 1;
    }
    // The 20 bits below the leading one, aligned as if the leading one sat at bit 31.
    let n = (x << (31 - l)) & 0xFFFF_FFFF;
    let sub = (n >> 26) & 31;
    let frac = (n >> 11) & 0x7FFF;
    let e = GAUSS_ICDF
        .get(usize::try_from((l << 5) | sub).unwrap_or(0))
        .map_or(0, |&e| i64::from(e));
    let base = e >> 12;
    let slope = e & 0xFFF;
    let mag = base - ((slope * frac + (1 << 14)) >> 15);
    if negative { -mag } else { mag }
}

/// The exact value `v` of one draw.
fn value(rng: &mut Pcg32, sigma_e: f32) -> f64 {
    (zq(rng.next_u32()) as f64) * TWO_POW_MINUS_17 * f64::from(sigma_e)
}

/// `v` rounded once to `f32`; `+0.0` when `sigma` counts as 0 (the product would keep the sign).
fn f32_of(v: f64, sigma_e: f32) -> f32 {
    if sigma_e == 0.0 { 0.0 } else { v as f32 }
}

/// Runs one call: `sample` makes one output sample from the generator and `sigma_e`.
fn run<T: Sample>(
    state: &mut GaussNoiseState,
    output: &mut [T],
    sample: impl Fn(&mut Pcg32, f32) -> T,
) -> Result<()> {
    let sigma_e = flush_denormal(state.sigma());
    let Some((rng, _)) = state.begin(T::TYPE, output.len())? else {
        return Ok(());
    };
    for o in output.iter_mut() {
        *o = sample(rng, sigma_e);
    }
    Ok(())
}

impl GaussNoise<f32> for Reference {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [f32]) -> Result<()> {
        run(state, output, |rng, s| f32_of(value(rng, s), s))
    }
}

impl GaussNoise<Q15> for Reference {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [Q15]) -> Result<()> {
        run(state, output, |rng, s| q15_from_f64(value(rng, s)))
    }
}

impl GaussNoise<C32> for Reference {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [C32]) -> Result<()> {
        run(state, output, |rng, s| {
            let re = f32_of(value(rng, s), s);
            let im = f32_of(value(rng, s), s);
            C32::new(re, im)
        })
    }
}

impl GaussNoise<CQ15> for Reference {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [CQ15]) -> Result<()> {
        run(state, output, |rng, s| {
            let re = q15_from_f64(value(rng, s));
            let im = q15_from_f64(value(rng, s));
            CQ15 { re, im }
        })
    }
}
