//! The SNR estimate of M44.4 §1: `snr_estimate_block` and `snr_estimate_block_svr`.
//!
//! | entry | measured | comparison | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `snr_estimate_block` | `SnrEstimateBlock::observe` | GNU Radio `mpsk_snr_est_cc` with `SNR_EST_M2M4` (`info`, reference does different work) | one input symbol |
//! | `snr_estimate_block_svr` | `SnrEstimateBlock::observe` | GNU Radio `mpsk_snr_est_cc` with `SNR_EST_SVR` (`info`, reference does different work) | one input symbol |
//!
//! The block runs `qam16` (its default) with a window of 4096 symbols on 4096 16-QAM symbols
//! per call at Es/N0 20 dB (unit power, complex Gaussian noise), the estimate carried from call
//! to call, through `observe` (the call of a composite: the work of the estimate without the copy
//! of `process`, whose library `memcpy` the counts of a `rep movsb` per byte overstate; GNU Radio
//! runs without its copy likewise); it has no kernel and no backend, so `ours` and `fallback` run
//! the same code. GNU
//! Radio's block runs the same symbols with `tag_nsamples` 4096 (an estimate per window, as
//! ours) and its default `alpha` 0.001. The work differs (D2504): GNU Radio's estimators average
//! moments of `|x|` (M2M4: `|x|^2` and `|x|^4`, SVR: `|x|^4` and `|x_n|^2 |x_(n-1)|^2`, with a
//! `cabsf` per `abs` of the source, six per symbol for M2M4 and two for SVR) and assume a PSK
//! constellation (kurtosis 1), so they read 16-QAM 13.5 dB low at 20 dB; ours decides every
//! symbol on the grid and sums the error, the decisions and the second and fourth powers. The
//! accuracy is the SNR of the estimate as a power ratio against the true Es/N0,
//! `10 log10(rho^2 / (rho_est - rho)^2)`.

use std::hint::black_box;

use cref_sys::GrMpskSnrEstCc;
use sdrk::{C32, Rate};
use sdrk_blocks::{ArenaStorage, Block, SnrEstimateBlock, SnrEstimateConfig};
use sdrk_core::kernels::constellation::Qam;

use super::{Impl, Kernel, Layout, Pcg32};

/// The window of ours and the `tag_nsamples` of GNU Radio.
const WINDOW: usize = 4096;
/// Es/N0 of the input in dB.
const ES_N0_DB: f64 = 20.0;
/// GNU Radio's default `alpha` of `mpsk_snr_est_cc`.
const GR_ALPHA: f64 = 0.001;

#[inline(never)]
fn ours_call(b: &mut SnrEstimateBlock<ArenaStorage<'static>>, x: &[C32]) {
    let _ = b.observe(black_box(x));
}

#[inline(never)]
fn gr_call(gr: &mut GrMpskSnrEstCc, x: &[[f32; 2]]) {
    let _ = gr.work(black_box(x));
}

fn block() -> SnrEstimateBlock<ArenaStorage<'static>> {
    let cfg = SnrEstimateConfig {
        window: WINDOW,
        ..SnrEstimateConfig::default()
    };
    SnrEstimateBlock::new(
        &cfg,
        Rate::hz(10_000).expect("rate"),
        &mut ArenaStorage::new(&mut []),
    )
    .expect("block")
}

/// `snr_estimate_block` (`svr` false) or `snr_estimate_block_svr` over `n` symbols per call.
pub(crate) struct SnrEstimateBlockK {
    svr: bool,
    ours: SnrEstimateBlock<ArenaStorage<'static>>,
    gr: Option<GrMpskSnrEstCc>,
    x: Vec<C32>,
    /// The symbols for GNU Radio: the history item (the last symbol) and then `x`.
    gr_in: Vec<[f32; 2]>,
}

impl SnrEstimateBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, svr: bool) -> Self {
        let q = Qam::Qam16;
        let sd = (10f64.powf(-ES_N0_DB / 10.0) / 2.0).sqrt();
        let mut unit = || (f64::from(rng.next_u32()) + 0.5) / 4_294_967_296.0;
        let x: Vec<C32> = (0..layout.n)
            .map(|_| {
                let p = q.point((unit() * 16.0) as u32);
                // Box-Muller
                let (u, v) = (unit(), unit());
                let r = (-2.0 * u.ln()).sqrt() * sd;
                let t = std::f64::consts::TAU * v;
                C32::new(
                    (f64::from(p.re) + r * t.cos()) as f32,
                    (f64::from(p.im) + r * t.sin()) as f32,
                )
            })
            .collect();
        let mut gr_in = Vec::with_capacity(layout.n + 1);
        let last = x.last().copied().unwrap_or_default();
        gr_in.push([last.re, last.im]);
        gr_in.extend(x.iter().map(|c| [c.re, c.im]));
        SnrEstimateBlockK {
            svr,
            ours: block(),
            gr: GrMpskSnrEstCc::new(svr, WINDOW, GR_ALPHA),
            x,
            gr_in,
        }
    }
}

impl Kernel for SnrEstimateBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => ours_call(&mut self.ours, &self.x),
                Impl::GnuRadio => gr_call(self.gr.as_mut().expect("gnuradio"), &self.gr_in),
                other => unreachable!("snr_estimate has no {other:?}"),
            }
            black_box(&self.ours);
        }
    }

    /// The SNR of the estimate after eight calls (as a power ratio) against the true Es/N0; 0
    /// when `process` does not pass the input through (ours) or the estimate is not finite.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let est = match which {
            Impl::GnuRadio => {
                self.gr = GrMpskSnrEstCc::new(self.svr, WINDOW, GR_ALPHA);
                self.run(which, 8);
                self.gr.as_ref().expect("gnuradio").last_snr()
            }
            _ => {
                self.ours = block();
                self.run(which, 7);
                let mut out = vec![C32::default(); self.x.len()];
                let _ = self.ours.process(&self.x, &mut out);
                if out != self.x {
                    return 0.0;
                }
                self.ours.snr_db()
            }
        };
        let rho = 10f64.powf(ES_N0_DB / 10.0);
        let got = 10f64.powf(est / 10.0);
        if !got.is_finite() {
            return 0.0;
        }
        (10.0 * (rho * rho / ((got - rho) * (got - rho))).log10()).clamp(0.0, 999.0)
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => self.gr.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.svr) {
            (Impl::Fallback, _) => "sdrk fallback",
            (Impl::GnuRadio, false) => {
                "GNU Radio mpsk_snr_est_cc(SNR_EST_M2M4, 4096, 0.001): the moments |x|^2 and |x|^4 averaged in double, six cabsf per symbol (reference does different work: no decisions, a PSK kurtosis; generic compiler flags)"
            }
            (Impl::GnuRadio, true) => {
                "GNU Radio mpsk_snr_est_cc(SNR_EST_SVR, 4096, 0.001): |x|^4 and |x_n|^2 |x_(n-1)|^2 averaged in double, two cabsf per symbol (reference does different work: no decisions, a PSK constellation; generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}
