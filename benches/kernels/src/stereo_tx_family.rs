//! The stereo transmit entries (milestone M36): `preemph_limit_stereo` (`C32`) and
//! `stereo_encode` (`C32` in, `f32` out).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `preemph_limit_stereo` | facade (= the fallback) | none (`info`) | our fallback |
//! | `stereo_encode` | facade (= the fallback) | none (`info`) | our fallback |
//!
//! Both kernels stay with the fallback on every SIMD backend (M36, "Not accelerated": the
//! limiter is a per-sample recursion, the encoder an oscillator whose two sines are table
//! lookups with interpolation at a running phase), so the facade entry is counted against our
//! own fallback as information at every level, as `preemph_limit` and `noise_blank`.
//!
//! The `preemph_limit_stereo` workload is that of `preemph_limit` with two channels: 48 kHz,
//! pre-emphasis of 50 us, limit 0.5, hold 20 ms, release 200 ms, quiet noise of 0.05 on both
//! channels (the limiter idles) with a 3 kHz burst of 0.9 on the left channel and a later 5 kHz
//! burst of 0.9 on the right channel, so that the limiter attacks on either channel and visits
//! attack, hold and release. The `stereo_encode` workload is the block default at 192 kHz (pilot
//! 19 kHz, `main` 0.9, `pilot` 0.09) on left 1 kHz 0.6 and right 3 kHz 0.4 (the tones of the
//! golden case `stereo_encode/c32_16384`) plus uniform noise of 0.01 on each channel.

use std::hint::black_box;

use sdrk::kernels::preemph_limit::{PreemphLimitStereo, PreemphLimitStereoState};
use sdrk::kernels::stereo::{StereoEncode, StereoEncodeState};
use sdrk::{C32, Fallback, Rate, Sdrk};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32s, f32s, snr_db};

/// `sin(2 pi f i / fs)` as f64.
fn tone(f: f64, fs: f64, i: usize) -> f64 {
    (2.0 * core::f64::consts::PI * f / fs * i as f64).sin()
}

// ---------------------------------------------------------------------------------------------
// preemph_limit_stereo

/// The sample rate of the `preemph_limit_stereo` workload.
const LIMIT_RATE_HZ: u64 = 48_000;

/// `preemph_limit_stereo` of `n` `C32` samples (left in `re`, right in `im`).
pub(crate) struct PreemphStereoK {
    input: Aligned<C32>,
    out: Aligned<C32>,
}

impl PreemphStereoK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let skew = layout.skew(core::mem::size_of::<C32>());
        let fs = LIMIT_RATE_HZ as f64;
        let mut input = Aligned::<C32>::skewed(n, skew);
        for (i, s) in input.iter_mut().enumerate() {
            let left = if (n / 8..n / 8 + n / 4).contains(&i) {
                0.9 * tone(3000.0, fs, i)
            } else {
                0.0
            };
            let right = if (5 * n / 8..11 * n / 16).contains(&i) {
                0.9 * tone(5000.0, fs, i)
            } else {
                0.0
            };
            let re = (left + f64::from(rng.unit()) * 0.05) as f32;
            let im = (right + f64::from(rng.unit()) * 0.05) as f32;
            *s = C32::new(re, im);
        }
        PreemphStereoK {
            input,
            out: Aligned::skewed(n, skew),
        }
    }

    fn state() -> PreemphLimitStereoState {
        let coefs = sdrk_blocks::design::preemphasis_coefs(50e-6, LIMIT_RATE_HZ as f64)
            .expect("preemphasis");
        PreemphLimitStereoState::new(
            coefs,
            0.5,
            0.02,
            0.2,
            Rate::hz(LIMIT_RATE_HZ).expect("rate"),
        )
        .expect("preemph_limit_stereo")
    }
}

#[inline(never)]
fn preemph_limit_stereo<B: PreemphLimitStereo>(
    backend: &B,
    state: &mut PreemphLimitStereoState,
    input: &[C32],
    out: &mut [C32],
) {
    let _ = black_box(backend.preemph_limit_stereo(state, black_box(input), out));
}

impl Kernel for PreemphStereoK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        for _ in 0..k {
            match which {
                Impl::Ours => preemph_limit_stereo(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => {
                    preemph_limit_stereo(&Fallback, &mut state, &self.input, &mut self.out)
                }
                other => unreachable!("preemph_limit_stereo has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![C32::new(0.0, 0.0); self.input.len()];
        let _ = Reference.preemph_limit_stereo(&mut Self::state(), &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&self.out))
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
// stereo_encode

/// The sample rate of the `stereo_encode` workload.
const ENCODE_RATE_HZ: u64 = 192_000;

/// `stereo_encode` of `n` `C32` samples (left in `re`, right in `im`) into `n` `f32` samples.
pub(crate) struct StereoEncodeK {
    input: Aligned<C32>,
    out: Aligned<f32>,
}

impl StereoEncodeK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let fs = ENCODE_RATE_HZ as f64;
        let mut input = Aligned::<C32>::skewed(n, layout.skew(core::mem::size_of::<C32>()));
        for (i, s) in input.iter_mut().enumerate() {
            let re = (0.6 * tone(1000.0, fs, i) + f64::from(rng.unit()) * 0.01) as f32;
            let im = (0.4 * tone(3000.0, fs, i) + f64::from(rng.unit()) * 0.01) as f32;
            *s = C32::new(re, im);
        }
        StereoEncodeK {
            input,
            out: Aligned::skewed(n, layout.skew(core::mem::size_of::<f32>())),
        }
    }

    fn state() -> StereoEncodeState {
        StereoEncodeState::new(19_000.0, 0.9, 0.09, Rate::hz(ENCODE_RATE_HZ).expect("rate"))
            .expect("stereo_encode")
    }
}

#[inline(never)]
fn stereo_encode<B: StereoEncode>(
    backend: &B,
    state: &mut StereoEncodeState,
    input: &[C32],
    out: &mut [f32],
) {
    let _ = black_box(backend.stereo_encode(state, black_box(input), out));
}

impl Kernel for StereoEncodeK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        for _ in 0..k {
            match which {
                Impl::Ours => stereo_encode(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => stereo_encode(&Fallback, &mut state, &self.input, &mut self.out),
                other => unreachable!("stereo_encode has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![0.0f32; self.input.len()];
        let _ = Reference.stereo_encode(&mut Self::state(), &self.input, &mut want);
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
