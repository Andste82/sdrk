//! The RDS transmit entries (milestone M42): `rds_shape` (bits to the `f32` RDS baseband) and
//! `stereo_encode_rds` (`C32` audio and the `f32` RDS baseband in, `f32` out).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `rds_shape` | facade (= the fallback) | none (`info`) | our fallback |
//! | `stereo_encode_rds` | facade (= the fallback) | none (`info`) | our fallback |
//!
//! Both kernels stay with the fallback on every SIMD backend (M42, "Not accelerated": the shaper
//! reads four interpolated table values per sample at an f64 position and fetches a bit through
//! a callback about every 162 samples, the encoder is `stereo_encode` plus a third table lookup),
//! so the facade entry is counted against our own fallback as information at every level, as
//! `stereo_encode`. Neither GNU Radio nor liquid-dsp has an RDS modulator in-tree, so there is no
//! library reference.
//!
//! The `rds_shape` workload is 192 kHz with random amplitudes +-1 from the generator (a stored
//! cycle of 4096 bits, fetched in order). The `stereo_encode_rds` workload is the block default
//! at 192 kHz (pilot 19 kHz, `main` 0.87, `pilot` 0.09, `rds` 0.04) on the audio of the
//! `stereo_encode` workload (left 1 kHz 0.6, right 3 kHz 0.4, each plus uniform noise of 0.01)
//! and the `rds_shape` output of random bits.

use std::hint::black_box;

use sdrk::kernels::rds::{RDS_TABLE_LEN, RdsShape, RdsShapeState, rds_pulse_table};
use sdrk::kernels::stereo::{StereoEncodeRds, StereoEncodeRdsState};
use sdrk::{C32, Fallback, Rate, Sdrk};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, f32s, snr_db};

/// The sample rate of both workloads.
const RATE_HZ: u64 = 192_000;
/// Bits in the cycle of the `rds_shape` workload.
const BITS: usize = 4096;

/// `sin(2 pi f i / fs)` as f64.
fn tone(f: f64, fs: f64, i: usize) -> f64 {
    (2.0 * core::f64::consts::PI * f / fs * i as f64).sin()
}

fn pulse_table() -> Vec<f32> {
    let mut t = vec![0.0f32; RDS_TABLE_LEN];
    rds_pulse_table(&mut t).expect("rds pulse table");
    t
}

fn random_bits(rng: &mut Pcg32) -> Vec<i8> {
    (0..BITS)
        .map(|_| if rng.unit() >= 0.0 { 1 } else { -1 })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// rds_shape

/// `rds_shape` of `n` `f32` samples.
pub(crate) struct RdsShapeK {
    table: Vec<f32>,
    bits: Vec<i8>,
    out: Aligned<f32>,
}

impl RdsShapeK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        RdsShapeK {
            table: pulse_table(),
            bits: random_bits(rng),
            out: Aligned::skewed(layout.n, layout.skew(core::mem::size_of::<f32>())),
        }
    }
}

#[inline(never)]
fn rds_shape<B: RdsShape>(
    backend: &B,
    state: &mut RdsShapeState<&[f32]>,
    bits: &mut dyn FnMut() -> i8,
    out: &mut [f32],
) {
    let _ = black_box(backend.rds_shape(state, bits, out));
}

/// Runs the shaper `k` times over `out` from a new state, with the bits of `bits` in a cycle.
fn shape_into<B: RdsShape>(backend: &B, table: &[f32], bits: &[i8], out: &mut [f32], k: usize) {
    let mut state = RdsShapeState::new(table, Rate::hz(RATE_HZ).expect("rate")).expect("rds_shape");
    let mut at = 0usize;
    let mut next = || {
        let b = bits.get(at % BITS).copied().unwrap_or(1);
        at += 1;
        b
    };
    for _ in 0..k {
        rds_shape(backend, &mut state, &mut next, out);
        black_box(&*out);
    }
}

impl Kernel for RdsShapeK {
    fn run(&mut self, which: Impl, k: usize) {
        match which {
            Impl::Ours => shape_into(&Sdrk, &self.table, &self.bits, &mut self.out, k),
            Impl::Fallback => shape_into(&Fallback, &self.table, &self.bits, &mut self.out, k),
            other => unreachable!("rds_shape has no {other:?}"),
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![0.0f32; self.out.len()];
        shape_into(&Reference, &self.table, &self.bits, &mut want, 1);
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

// ---------------------------------------------------------------------------------------------
// stereo_encode_rds

/// `stereo_encode_rds` of `n` `C32` and `n` `f32` samples into `n` `f32` samples.
pub(crate) struct StereoEncodeRdsK {
    input: Aligned<C32>,
    rds: Aligned<f32>,
    out: Aligned<f32>,
}

impl StereoEncodeRdsK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let fs = RATE_HZ as f64;
        let mut input = Aligned::<C32>::skewed(n, layout.skew(core::mem::size_of::<C32>()));
        for (i, s) in input.iter_mut().enumerate() {
            let re = (0.6 * tone(1000.0, fs, i) + f64::from(rng.unit()) * 0.01) as f32;
            let im = (0.4 * tone(3000.0, fs, i) + f64::from(rng.unit()) * 0.01) as f32;
            *s = C32::new(re, im);
        }
        let skew = layout.skew(core::mem::size_of::<f32>());
        let mut rds = Aligned::<f32>::skewed(n, skew);
        let (table, bits) = (pulse_table(), random_bits(rng));
        shape_into(&Fallback, &table, &bits, &mut rds, 1);
        StereoEncodeRdsK {
            input,
            rds,
            out: Aligned::skewed(n, skew),
        }
    }

    fn state() -> StereoEncodeRdsState {
        StereoEncodeRdsState::new(19_000.0, 0.87, 0.09, 0.04, Rate::hz(RATE_HZ).expect("rate"))
            .expect("stereo_encode_rds")
    }
}

#[inline(never)]
fn stereo_encode_rds<B: StereoEncodeRds>(
    backend: &B,
    state: &mut StereoEncodeRdsState,
    input: &[C32],
    rds: &[f32],
    out: &mut [f32],
) {
    let _ = black_box(backend.stereo_encode_rds(state, black_box(input), black_box(rds), out));
}

impl Kernel for StereoEncodeRdsK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        for _ in 0..k {
            match which {
                Impl::Ours => {
                    stereo_encode_rds(&Sdrk, &mut state, &self.input, &self.rds, &mut self.out);
                }
                Impl::Fallback => {
                    stereo_encode_rds(&Fallback, &mut state, &self.input, &self.rds, &mut self.out);
                }
                other => unreachable!("stereo_encode_rds has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let mut want = vec![0.0f32; self.input.len()];
        let _ = Reference.stereo_encode_rds(&mut Self::state(), &self.input, &self.rds, &mut want);
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
