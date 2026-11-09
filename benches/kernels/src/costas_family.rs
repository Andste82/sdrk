//! The carrier recovery entries (milestone M40): `costas` and `costas_fallback` (`C32`).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `costas` | facade (= the fallback) | none (`info`) | our fallback |
//! | `costas_fallback` | our fallback | GNU Radio's `costas_loop_cc` | — |
//!
//! The kernel stays with the fallback on every SIMD backend (M40, "Not accelerated": the phase
//! of every symbol depends on the detector output of the symbol before it), so the facade entry
//! is counted against our own fallback as information, as `symbol_sync`. The fallback entry is
//! counted against the loop of GNU Radio's `costas_loop_cc` (`benches/refs/gr-costas.c`, D607):
//! the same second-order loop with the same detectors, with `sincosf` instead of the table
//! lookup, a clip of the error to `+-1` and no lock detector.
//!
//! The workload is QPSK with one sample per symbol (the input of the kernel after
//! `symbol_sync`), a start phase of 0.7 rad, a carrier offset of 0.002 cycles per symbol and
//! uniform noise of 0.15 per component, with the block defaults (order 4, `loop_bw` 0.02,
//! `damping` 0.7071, `max_offset` 0.05); GNU Radio runs order 4 with the loop bandwidth
//! `theta = loop_bw / (damping + 1 / (4 damping))` of our loop constants, so both loops have
//! the same gains (GNU Radio's in radians, ours in cycles). A "sample" is a symbol.

use std::hint::black_box;

use cref_sys::GrCostasLoopCc;
use sdrk::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk::{C32, Fallback, Sdrk};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32_pairs, c32_pairs_ref, c32s, snr_db};

/// The block defaults of `costas` (D604).
const ORDER: u32 = 4;
/// The constellation of [`ORDER`].
const CONSTELLATION: CostasConstellation = CostasConstellation::Qpsk;
const LOOP_BW: f64 = 0.02;
#[allow(clippy::approx_constant)]
const DAMPING: f64 = 0.7071;
const MAX_OFFSET: f64 = 0.05;
/// The start phase (rad), the carrier offset (cycles per symbol) and the noise per component.
const PHASE0: f64 = 0.7;
const OFFSET: f64 = 0.002;
const NOISE: f64 = 0.15;

/// The loop bandwidth of GNU Radio's `control_loop` that gives our loop constants.
fn gr_loop_bw() -> f32 {
    (LOOP_BW / (DAMPING + 1.0 / (4.0 * DAMPING))) as f32
}

fn gr_costas() -> Option<GrCostasLoopCc> {
    GrCostasLoopCc::new(gr_loop_bw(), ORDER)
}

/// The f64 model of the `costas_loop_cc` loop (exact sine and cosine), the accuracy reference
/// of [`Impl::GnuRadio`].
fn gr_costas_model(input: &[C32]) -> Vec<C32> {
    let Some(cl) = gr_costas() else {
        return Vec::new();
    };
    let st = cl.state();
    let (alpha, beta) = (f64::from(st.alpha), f64::from(st.beta));
    let sign = |v: f64| if v > 0.0 { 1.0 } else { -1.0 };
    let (mut phase, mut freq) = (0.0f64, 0.0f64);
    let mut out = Vec::with_capacity(input.len());
    for x in input {
        let (s, c) = (-phase).sin_cos();
        let (xr, xi) = (f64::from(x.re), f64::from(x.im));
        let (yr, yi) = (xr * c - xi * s, xr * s + xi * c);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a model value rounded to f32"
        )]
        out.push(C32::new(yr as f32, yi as f32));
        let e = (sign(yr) * yi - sign(yi) * yr).clamp(-1.0, 1.0);
        freq += beta * e;
        phase += freq + alpha * e;
        let two_pi = 2.0 * core::f64::consts::PI;
        while phase > two_pi {
            phase -= two_pi;
        }
        while phase < -two_pi {
            phase += two_pi;
        }
        freq = freq.clamp(-1.0, 1.0);
    }
    out
}

/// `costas` of `n` symbols.
pub(crate) struct CostasK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    /// The entry: `costas_fallback`.
    fallback_entry: bool,
}

impl CostasK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, fallback_entry: bool) -> Self {
        let n = layout.n;
        let skew = layout.skew(core::mem::size_of::<C32>());
        let r = core::f64::consts::FRAC_1_SQRT_2;
        let mut input = Aligned::<C32>::skewed(n, skew);
        for (k, s) in input.iter_mut().enumerate() {
            let b = rng.next_u32();
            let (re, im) = (
                if b & 1 == 1 { r } else { -r },
                if b & 2 == 2 { r } else { -r },
            );
            let turns = (PHASE0 / core::f64::consts::TAU + OFFSET * k as f64).fract();
            let (sn, c) = (core::f64::consts::TAU * turns).sin_cos();
            let nr = f64::from(rng.unit()) * NOISE;
            let ni = f64::from(rng.unit()) * NOISE;
            #[allow(clippy::cast_possible_truncation, reason = "the f32 input")]
            {
                *s = C32::new(
                    (re * c - im * sn + nr) as f32,
                    (re * sn + im * c + ni) as f32,
                );
            }
        }
        CostasK {
            input,
            out: Aligned::skewed(n, skew),
            fallback_entry,
        }
    }

    fn state() -> CostasState {
        CostasState::new(CONSTELLATION, LOOP_BW, DAMPING, MAX_OFFSET).expect("costas")
    }
}

#[inline(never)]
fn costas<B: Costas>(backend: &B, state: &mut CostasState, input: &[C32], out: &mut [C32]) {
    let _ = black_box(backend.costas(state, black_box(input), out));
}

impl Kernel for CostasK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = Self::state();
        let mut gr = gr_costas();
        for _ in 0..k {
            match which {
                Impl::Ours => costas(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => costas(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::GnuRadio => gr
                    .as_mut()
                    .expect("gnuradio")
                    .work(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                other => unreachable!("costas has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = if which == Impl::GnuRadio {
            gr_costas_model(&self.input)
        } else {
            let mut want = vec![C32::default(); self.input.len()];
            let _ = Reference.costas(&mut Self::state(), &self.input, &mut want);
            want
        };
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours => !self.fallback_entry,
            Impl::Fallback => true,
            Impl::GnuRadio => self.fallback_entry && gr_costas().is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio costas_loop_cc (costas_loop_cc_impl.cc of gr-digital, generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}
