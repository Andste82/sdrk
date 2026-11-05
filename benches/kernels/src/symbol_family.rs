//! The symbol timing recovery entries (milestone M38): `symbol_sync` (`C32`) and
//! `symbol_sync_real` (`f32`).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `symbol_sync` | facade (= the fallback) | none (`info`) | our fallback |
//! | `symbol_sync_real` | facade (= the fallback) | none (`info`) | our fallback |
//!
//! The kernel stays with the fallback on every SIMD backend (M38, "Not accelerated": every
//! strobe position depends on the timing error of the strobe before it), so the facade entry is
//! counted against our own fallback as information at every level, as `noise_blank` and
//! `preemph_limit_stereo`. No library runs the same loop: GNU Radio's `symbol_sync_cc`/`_ff`
//! interpolate with an 8-tap MMSE filter bank or a polyphase filter bank, liquid-dsp's
//! `symsync` with a polyphase matched filter bank and its derivative, while `symbol_sync`
//! interpolates with a 4-point cubic Lagrange polynomial (D562).
//!
//! The workloads are the shapes of the golden cases with the block defaults (Gardner, `loop_bw`
//! 0.01, `damping` 0.7071, `max_deviation` 0.005, `ted_gain` 1.0): `symbol_sync` runs BPSK
//! with a raised-cosine pulse (roll-off 0.35, 8 symbols per side) at 4.1 samples per symbol,
//! `symbol_sync_real` real `+-1` levels at 8 samples per symbol through a boxcar of one symbol
//! (the shape of a 2-FSK discriminator output); both with uniform noise of 0.05 per component.
//! A "sample" is an input sample; the cost per symbol is the count times the samples per
//! symbol.

use std::hint::black_box;

use sdrk::kernels::symbol_sync::{SymbolSync, SymbolSyncState, Ted};
use sdrk::{C32, Fallback, Sample, Sdrk};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32s, f32s, snr_db};

/// The block defaults of `symbol_sync` (M38 §2, D550).
const LOOP_BW: f64 = 0.01;
#[allow(clippy::approx_constant)]
const DAMPING: f64 = 0.7071;
const MAX_DEVIATION: f64 = 0.005;
const TED_GAIN: f64 = 1.0;

/// Samples per symbol of the `C32` workload.
const SPS_C32: f64 = 4.1;
/// Samples per symbol of the `f32` workload.
const SPS_F32: usize = 8;
/// Roll-off of the raised-cosine pulse.
const ROLLOFF: f64 = 0.35;
/// Truncation of the pulse in symbols per side.
const SPAN: f64 = 8.0;

/// The raised-cosine pulse at `t` symbols.
fn raised_cosine(t: f64) -> f64 {
    use core::f64::consts::PI;
    if t.abs() < 1e-9 {
        return 1.0;
    }
    let sinc = (PI * t).sin() / (PI * t);
    let den = 1.0 - (2.0 * ROLLOFF * t) * (2.0 * ROLLOFF * t);
    if den.abs() < 1e-9 {
        // The limit at t = +-1 / (2 b).
        return PI / 4.0 * sinc;
    }
    sinc * (PI * ROLLOFF * t).cos() / den
}

/// A random `+-1`.
fn level(rng: &mut Pcg32) -> f64 {
    if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 }
}

/// `symbol_sync` of `n` input samples of type `T` (`C32` or `f32`).
pub(crate) struct SymbolSyncK<T: Sample> {
    sps: f64,
    input: Aligned<T>,
    out: Aligned<T>,
    produced: usize,
}

impl SymbolSyncK<C32> {
    /// The `C32` workload: raised-cosine BPSK at 4.1 samples per symbol.
    pub(crate) fn c32(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let skew = layout.skew(core::mem::size_of::<C32>());
        let symbols = (n as f64 / SPS_C32).ceil() as usize + 2 * SPAN as usize + 1;
        let bits: Vec<f64> = (0..symbols).map(|_| level(rng)).collect();
        let mut input = Aligned::<C32>::skewed(n, skew);
        for (i, s) in input.iter_mut().enumerate() {
            // Symbol `k` sits at the sample `(k - SPAN) sps`.
            let t = i as f64 / SPS_C32 + SPAN;
            let lo = (t - SPAN).ceil().max(0.0) as usize;
            let hi = ((t + SPAN).floor() as usize).min(symbols - 1);
            let v: f64 = (lo..=hi)
                .map(|k| bits[k] * raised_cosine(t - k as f64))
                .sum();
            let re = (v + f64::from(rng.unit()) * 0.05) as f32;
            let im = (f64::from(rng.unit()) * 0.05) as f32;
            *s = C32::new(re, im);
        }
        SymbolSyncK {
            sps: SPS_C32,
            input,
            out: Aligned::skewed(n, skew),
            produced: 0,
        }
    }
}

impl SymbolSyncK<f32> {
    /// The `f32` workload: boxcar-filtered `+-1` levels at 8 samples per symbol.
    pub(crate) fn f32(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let skew = layout.skew(core::mem::size_of::<f32>());
        let levels: Vec<f64> = (0..n.div_ceil(SPS_F32) + 1).map(|_| level(rng)).collect();
        let mut input = Aligned::<f32>::skewed(n, skew);
        for (i, s) in input.iter_mut().enumerate() {
            // The mean of the levels over the last `SPS_F32` samples (`i - 7 ..= i`).
            let v: f64 = (0..SPS_F32)
                .map(|j| {
                    let p = (i + SPS_F32 - j) / SPS_F32;
                    levels[p]
                })
                .sum::<f64>()
                / SPS_F32 as f64;
            *s = (v + f64::from(rng.unit()) * 0.05) as f32;
        }
        SymbolSyncK {
            sps: SPS_F32 as f64,
            input,
            out: Aligned::skewed(n, skew),
            produced: 0,
        }
    }
}

#[inline(never)]
fn symbol_sync<T: Sample, B: SymbolSync<T>>(
    backend: &B,
    state: &mut SymbolSyncState<T>,
    input: &[T],
    out: &mut [T],
) -> usize {
    black_box(backend.symbol_sync(state, black_box(input), out)).map_or(0, |p| p.produced)
}

impl<T: Sample> SymbolSyncK<T>
where
    Sdrk: SymbolSync<T>,
    Fallback: SymbolSync<T>,
    Reference: SymbolSync<T>,
{
    fn state(&self) -> SymbolSyncState<T> {
        SymbolSyncState::new(
            self.sps,
            Ted::Gardner,
            LOOP_BW,
            DAMPING,
            MAX_DEVIATION,
            TED_GAIN,
        )
        .expect("symbol_sync")
    }

    fn run_impl(&mut self, which: Impl, k: usize) {
        let mut state = self.state();
        for _ in 0..k {
            self.produced = match which {
                Impl::Ours => symbol_sync(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => symbol_sync(&Fallback, &mut state, &self.input, &mut self.out),
                other => unreachable!("symbol_sync has no {other:?}"),
            };
            black_box(&self.out);
        }
    }

    /// The outputs of one run of `Reference` (the exact float32 model).
    fn reference(&self) -> Vec<T> {
        let mut want = vec![T::default(); self.input.len()];
        let mut state = self.state();
        let produced = Reference
            .symbol_sync(&mut state, &self.input, &mut want)
            .map_or(0, |p| p.produced);
        want.truncate(produced);
        want
    }
}

impl Kernel for SymbolSyncK<C32> {
    fn run(&mut self, which: Impl, k: usize) {
        self.run_impl(which, k);
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = self.reference();
        if self.produced == 0 || want.len() != self.produced {
            return f64::NEG_INFINITY;
        }
        snr_db(&c32s(&want), &c32s(&self.out[..self.produced]))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}

impl Kernel for SymbolSyncK<f32> {
    fn run(&mut self, which: Impl, k: usize) {
        self.run_impl(which, k);
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = self.reference();
        if self.produced == 0 || want.len() != self.produced {
            return f64::NEG_INFINITY;
        }
        snr_db(&f32s(&want), &f32s(&self.out[..self.produced]))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            _ => "sdrk",
        }
    }
}
