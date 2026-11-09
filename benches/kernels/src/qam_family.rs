//! The QAM entries of M44: `slice_qam16_block` and `slice_qam64_block`.
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `slice_qam16_block` | `sdrk_blocks::SliceBlock` with `qam16` | none (`info`) | one input symbol (four soft bits) |
//! | `slice_qam64_block` | `sdrk_blocks::SliceBlock` with `qam64` | none (`info`) | one input symbol (six soft bits) |
//!
//! The slicer has no kernel: per symbol it puts both components on the integer grid (`v = x √E`)
//! and forms the piecewise-linear max-log soft bits of M44 §2 (`v`, `2 - |v|`, and for 64-QAM
//! `4 - |v|`, `2 - ||v| - 4|`), branch-free arithmetic that the compiler vectorizes where it
//! can. No library computes the same soft bits: liquid-dsp's `modemcf_demodulate_soft` and GNU
//! Radio's `calc_soft_dec` search the points for the exact max-log or log-sum value (D981), so
//! both entries are our own code as information (`Family::Own`), and `ours` and `fallback` run
//! the same code.
//!
//! Workload: 4096 symbols per call, random points of the constellation with uniform noise of
//! ±0.05 per component, scale 1.

use std::hint::black_box;

use sdrk::{C32, Rate};
use sdrk_blocks::{ArenaStorage, Block, SliceBlock, SliceConfig, SliceConstellation};
use sdrk_core::kernels::constellation::Qam;

use super::{Impl, Kernel, Layout, Pcg32, snr_db};

/// The block `slice` with a QAM constellation over `n` symbols per call.
pub(crate) struct SliceQamBlockK {
    qam: Qam,
    block: SliceBlock<C32, ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<f32>,
}

impl SliceQamBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam) -> Self {
        let constellation = match qam {
            Qam::Qam16 => SliceConstellation::Qam16,
            _ => SliceConstellation::Qam64,
        };
        let cfg = SliceConfig {
            constellation,
            scale: 1.0,
        };
        let block = SliceBlock::new(
            &cfg,
            Rate::hz(25_000).expect("rate"),
            &mut ArenaStorage::new(&mut []),
        )
        .expect("slice block");
        let input = (0..layout.n)
            .map(|_| {
                let p = qam.point(rng.next_u32());
                C32::new(p.re + 0.05 * rng.unit(), p.im + 0.05 * rng.unit())
            })
            .collect();
        SliceQamBlockK {
            qam,
            block,
            input,
            out: vec![0.0; layout.n * qam.bits_per_symbol()],
        }
    }
}

#[inline(never)]
fn slice_call(block: &mut SliceBlock<C32, ArenaStorage<'static>>, x: &[C32], out: &mut [f32]) {
    let _ = block.process(black_box(x), out);
}

/// The soft bits of one axis value in `f64`: `v = x √E`, then `v`, `2 - |v|` (16-QAM) or `v`,
/// `4 - |v|`, `2 - ||v| - 4|` (64-QAM).
fn axis_f64(qam: Qam, x: f32, out: &mut Vec<f64>) {
    let v = f64::from(x) * qam.energy().sqrt();
    let a = v.abs();
    out.push(v);
    match qam {
        Qam::Qam16 => out.push(2.0 - a),
        _ => {
            out.push(4.0 - a);
            out.push(2.0 - (a - 4.0).abs());
        }
    }
}

impl Kernel for SliceQamBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    slice_call(&mut self.block, &self.input, &mut self.out);
                }
                other => unreachable!("slice_qam has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The soft bits against the definition of M44 §2 computed in `f64`.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        slice_call(&mut self.block, &self.input, &mut self.out);
        let mut want = Vec::with_capacity(self.out.len());
        for y in &self.input {
            axis_f64(self.qam, y.re, &mut want);
            axis_f64(self.qam, y.im, &mut want);
        }
        let got: Vec<f64> = self.out.iter().map(|&v| f64::from(v)).collect();
        snr_db(&want, &got)
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
