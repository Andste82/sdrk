//! Fallback implementation of the `gauss_noise` kernel.
//!
//! The integer path of the contract: one draw per real value, [`normal_q17`] and the output
//! step of [`GaussScale`]. A complex buffer of `n` samples is the real kernel on its `2n` values
//! in draw order (re, im, …), so there is one inner loop per output format. The draws come from
//! two interleaved generator chains ([`Pcg32::fill_interleaved`]): the same draws and the same
//! final state as one chain, but two independent multiply chains, which halves the latency
//! bound of the LCG. With `sigma` counting as 0 the generator jumps over the draws of the call in
//! `O(log n)` steps instead of drawing.

use crate::Fallback;
use crate::error::Result;
use crate::kernels::noise::{GaussNoise, GaussNoiseState, GaussScale, Pcg32, normal_q17};
use crate::types::{C32, CQ15, Q15, Sample};

/// Interleaved generator chains of the fallback.
const CHAINS: usize = 2;

/// Runs one call over the real values `values` of an output of `T`: fills zeros and jumps the
/// generator for a zero scale, otherwise maps every draw with `value`.
#[inline]
fn run<T: Sample, V: Copy + Default>(
    state: &mut GaussNoiseState,
    len: usize,
    values: &mut [V],
    value: impl Fn(GaussScale, i32) -> V,
) -> Result<()> {
    let Some((rng, scale)) = state.begin(T::TYPE, len)? else {
        return Ok(());
    };
    fill(rng, scale, values, value);
    Ok(())
}

#[inline]
fn fill<V: Copy + Default>(
    rng: &mut Pcg32,
    scale: GaussScale,
    values: &mut [V],
    value: impl Fn(GaussScale, i32) -> V,
) {
    if scale.is_zero() {
        // `V::default()` is `+0.0` or `Q15(0)`.
        values.fill(V::default());
        rng.advance(values.len() as u64);
        return;
    }
    rng.map_interleaved::<CHAINS, V>(values, |r| value(scale, normal_q17(r)));
}

fn f32_of(scale: GaussScale, zq: i32) -> f32 {
    scale.f32_value(zq)
}

fn q15_of(scale: GaussScale, zq: i32) -> Q15 {
    scale.q15_value(zq)
}

impl GaussNoise<f32> for Fallback {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [f32]) -> Result<()> {
        run::<f32, f32>(state, output.len(), output, f32_of)
    }
}

impl GaussNoise<Q15> for Fallback {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [Q15]) -> Result<()> {
        run::<Q15, Q15>(state, output.len(), output, q15_of)
    }
}

impl GaussNoise<C32> for Fallback {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [C32]) -> Result<()> {
        let len = output.len();
        let values: &mut [f32] = bytemuck::cast_slice_mut(output);
        run::<C32, f32>(state, len, values, f32_of)
    }
}

impl GaussNoise<CQ15> for Fallback {
    fn gauss_noise(&self, state: &mut GaussNoiseState, output: &mut [CQ15]) -> Result<()> {
        let len = output.len();
        let values: &mut [Q15] = bytemuck::cast_slice_mut(output);
        run::<CQ15, Q15>(state, len, values, q15_of)
    }
}
