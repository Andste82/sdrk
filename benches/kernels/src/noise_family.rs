//! The noise reduction entries (milestone M33): `noise_blank` (`C32`), `noise_blank_cq15`
//! (`CQ15`) and `auto_notch` (`f32`).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `noise_blank` | facade (= the fallback) | none (`info`) | our fallback |
//! | `noise_blank_cq15` | facade (= the fallback) | none (`info`) | our fallback |
//! | `auto_notch` | facade (= the fallback) | none (`info`) | our fallback |
//!
//! Both kernels are per-sample recursions and stay with the fallback on every SIMD backend
//! (M33 Tests 5), so the facade entry is counted against our own fallback as information at
//! every level, as `squelch_cq15` and `preemph_limit`.
//!
//! The `noise_blank` workload is the block default at 192 kHz (threshold 15 dB, time
//! constant 10 ms, hold 0.2 ms = 38 samples, lookahead 50 us = 10 samples): complex noise of
//! 0.05 per component with an impulse of 3 samples and amplitude 0.8 at a random phase in every
//! slot of 512 samples, so every call takes both paths (pass through and blanking; about one
//! output sample in ten is blanked). The `auto_notch` workload is the block default (32 taps,
//! delay 8, step size 0.01) on audio at 12 kHz, the rate of the flow `hf_am_receiver`: a
//! whistle of 1.5 kHz with amplitude 0.3 plus uniform noise of 0.1.

use std::hint::black_box;

use sdrk::kernels::auto_notch::{AutoNotch, AutoNotchState};
use sdrk::kernels::noise_blank::{NoiseBlank, NoiseBlankState};
use sdrk::{C32, CQ15, ComplexSample, Fallback, Q15, Rate, Sdrk};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32s, cq15s, f32s, snr_db};

// ---------------------------------------------------------------------------------------------
// noise_blank

/// The sample rate of the `noise_blank` workload.
const BLANK_RATE_HZ: u64 = 192_000;
/// The lookahead `D = round(50 us * 192 kHz)` of the block default, in samples.
const BLANK_LOOKAHEAD: usize = 10;
/// One impulse per slot of this many samples.
const BLANK_SLOT: usize = 512;

/// A complex sample of the `noise_blank` measurements.
pub(crate) trait BlankSample: ComplexSample {
    /// The sample `(re, im)` given as fractions of full scale.
    fn from_f32(re: f32, im: f32) -> Self;
    /// The components as f64 (raw integers for `CQ15`).
    fn flat(x: &[Self]) -> Vec<f64>;
}

impl BlankSample for C32 {
    fn from_f32(re: f32, im: f32) -> Self {
        C32::new(re, im)
    }

    fn flat(x: &[Self]) -> Vec<f64> {
        c32s(x)
    }
}

impl BlankSample for CQ15 {
    fn from_f32(re: f32, im: f32) -> Self {
        let q = |v: f32| Q15((v * 32768.0).round().clamp(-32768.0, 32767.0) as i16);
        CQ15 {
            re: q(re),
            im: q(im),
        }
    }

    fn flat(x: &[Self]) -> Vec<f64> {
        cq15s(x, 1.0)
    }
}

/// `noise_blank` of `n` samples with the block defaults at 192 kHz.
pub(crate) struct NoiseBlankK<T: BlankSample> {
    input: Aligned<T>,
    out: Aligned<T>,
}

impl<T: BlankSample> NoiseBlankK<T> {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let skew = layout.skew(core::mem::size_of::<T>());
        let mut input = Aligned::<T>::skewed(n, skew);
        let mut impulse = (0usize, 0.0f32);
        for (i, s) in input.iter_mut().enumerate() {
            if i % BLANK_SLOT == 0 {
                // The start of the impulse in this slot (away from the slot edges) and its phase.
                let start = i + 16 + (rng.next_u32() as usize) % (BLANK_SLOT - 32);
                impulse = (start, rng.unit() * core::f32::consts::PI);
            }
            let (mut re, mut im) = (rng.unit() * 0.05, rng.unit() * 0.05);
            if (impulse.0..impulse.0 + 3).contains(&i) {
                re += 0.8 * impulse.1.cos();
                im += 0.8 * impulse.1.sin();
            }
            *s = T::from_f32(re, im);
        }
        NoiseBlankK {
            input,
            out: Aligned::skewed(n, skew),
        }
    }

    fn state() -> NoiseBlankState<T, [T; BLANK_LOOKAHEAD]> {
        NoiseBlankState::new(
            15.0,
            0.01,
            0.0002,
            [T::ZERO; BLANK_LOOKAHEAD],
            Rate::hz(BLANK_RATE_HZ).expect("rate"),
        )
        .expect("noise_blank")
    }
}

#[inline(never)]
fn noise_blank<T: ComplexSample, B: NoiseBlank<T>>(
    backend: &B,
    state: &mut NoiseBlankState<T, [T; BLANK_LOOKAHEAD]>,
    input: &[T],
    out: &mut [T],
) {
    let _ = black_box(backend.noise_blank(state, black_box(input), out));
}

impl<T: BlankSample> Kernel for NoiseBlankK<T>
where
    Sdrk: NoiseBlank<T>,
    Fallback: NoiseBlank<T>,
    Reference: NoiseBlank<T>,
{
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        for _ in 0..k {
            match which {
                Impl::Ours => noise_blank(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => noise_blank(&Fallback, &mut state, &self.input, &mut self.out),
                other => unreachable!("noise_blank has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![T::ZERO; self.input.len()];
        let _ = Reference.noise_blank(&mut Self::state(), &self.input, &mut want);
        snr_db(&T::flat(&want), &T::flat(&self.out))
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

// ---------------------------------------------------------------------------------------------
// auto_notch

/// The taps `L` of the block default.
const NOTCH_TAPS: usize = 32;
/// The delay `Δ` of the block default.
const NOTCH_DELAY: usize = 8;

/// `auto_notch` of `n` `f32` samples with the block defaults on 12 kHz audio.
pub(crate) struct AutoNotchK {
    input: Aligned<f32>,
    out: Aligned<f32>,
}

impl AutoNotchK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let skew = layout.skew(core::mem::size_of::<f32>());
        let mut input = Aligned::<f32>::skewed(n, skew);
        for (i, s) in input.iter_mut().enumerate() {
            let whistle = 0.3 * (2.0 * core::f64::consts::PI * 1500.0 / 12_000.0 * i as f64).sin();
            *s = (whistle + f64::from(rng.unit()) * 0.1) as f32;
        }
        AutoNotchK {
            input,
            out: Aligned::skewed(n, skew),
        }
    }

    fn state() -> AutoNotchState<[f32; NOTCH_TAPS], [f32; NOTCH_TAPS + NOTCH_DELAY]> {
        AutoNotchState::new(
            [0.0; NOTCH_TAPS],
            [0.0; NOTCH_TAPS + NOTCH_DELAY],
            NOTCH_DELAY,
            0.01,
        )
        .expect("auto_notch")
    }
}

#[inline(never)]
fn auto_notch<B: AutoNotch>(
    backend: &B,
    state: &mut AutoNotchState<[f32; NOTCH_TAPS], [f32; NOTCH_TAPS + NOTCH_DELAY]>,
    input: &[f32],
    out: &mut [f32],
) {
    let _ = black_box(backend.auto_notch(state, black_box(input), out));
}

impl Kernel for AutoNotchK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        for _ in 0..k {
            match which {
                Impl::Ours => auto_notch(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => auto_notch(&Fallback, &mut state, &self.input, &mut self.out),
                other => unreachable!("auto_notch has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![0.0f32; self.input.len()];
        let _ = Reference.auto_notch(&mut Self::state(), &self.input, &mut want);
        snr_db(&f32s(&want), &f32s(&self.out))
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
