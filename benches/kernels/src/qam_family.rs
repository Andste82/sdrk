//! The QAM entries of M44: `qam_soft_bits_qam16`, `qam_soft_bits_qam64`, `slice_qam16_block`,
//! `slice_qam64_block`, `costas_qam16`, `costas_qam16_fallback`, `equalize_qam16`,
//! `equalize_qam16_fallback`, `qam_mod_block`, `qam_demod_block` and `frame_decode_four_block`.
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `qam_soft_bits_qam16` | the kernel `qam_soft_bits` with `Qam16` through the facade | our fallback (`info`) | one input symbol (four soft bits) |
//! | `qam_soft_bits_qam64` | the kernel `qam_soft_bits` with `Qam64` through the facade | our fallback (`info`) | one input symbol (six soft bits) |
//! | `slice_qam16_block` | `sdrk_blocks::SliceBlock` with `qam16` | none (`info`) | one input symbol (four soft bits) |
//! | `slice_qam64_block` | `sdrk_blocks::SliceBlock` with `qam64` | none (`info`) | one input symbol (six soft bits) |
//! | `costas_qam16` | facade (= the fallback) | none (`info`) | one symbol |
//! | `costas_qam16_fallback` | our fallback | GNU Radio's `constellation_receiver_cb` | one symbol |
//! | `equalize_qam16` | facade (= the fallback's loop, D841) | none (`info`) | one symbol |
//! | `equalize_qam16_fallback` | our fallback | GNU Radio's `linear_equalizer` on a `constellation_rect` | one symbol |
//! | `qam_mod_block` | `sdrk_blocks::QamModBlock`, 16-QAM | none (`info`) | one output sample |
//! | `qam_demod_block` | `sdrk_blocks::QamDemodBlock`, 16-QAM | none (`info`) | one input sample |
//! | `frame_decode_four_block` | `sdrk_blocks::FrameDecodeBlock` with `rotations` `four` | none (`info`) | one input sample (one soft bit) |
//!
//! The kernel `qam_soft_bits` (I44-1, D1412) puts both components of every symbol on the
//! integer grid (`v = x √E`) and forms the piecewise-linear max-log soft bits of M44 §2 (`v`,
//! `2 - |v|`, and for 64-QAM `4 - |v|`, `2 - ||v| - 4|`); the SIMD backend has a hand-written
//! path per level (D1414), the fallback is the component loop that the compiler vectorizes
//! where it can. No library computes the same soft bits: liquid-dsp's `modemcf_demodulate_soft`
//! and GNU Radio's `calc_soft_dec` search the points for the exact max-log or log-sum value
//! (D1201), so the kernel entries are our own code, judged against our fallback as information
//! (`Family::Own`). The block `slice` calls the kernel through the facade; its entries
//! (`Family::Own` as well, `ours` and `fallback` the same block code) measure the kernel with
//! the block around it.
//!
//! The blocks `qam_mod` and `qam_demod` (M44 §4) are chains of kernels that have their own
//! entries with their library references (`fir_interpolate`, `fir`, `symbol_sync`,
//! `equalize_qam16_fallback` and `costas_qam16_fallback` against GNU Radio, the slicer above); no
//! library has the same block (liquid-dsp's `modemcf` maps and slices but has no pulse shaping,
//! timing, equalizer or carrier loop in the same object, GNU Radio's receiver is a flowgraph of
//! blocks), so the block entries are our own code as information (D1216), as `channelize_block`
//! (D928). `qam_mod_block` is 16-QAM at 8 samples per symbol (the flows of M44 §5) writing 4096
//! output samples (512 symbols) per call; `qam_demod_block` reads 4096 input samples of that
//! signal per call with the block defaults after a warm-up of 3072 symbols (the equalizer in
//! decision mode, the loops locked). `frame_decode_four_block` is the decoder of
//! `frame_decode_block` with `rotations` `four` and 4 bits per symbol on frames turned by one
//! quarter turn: the cost of the four inverse rotations and sync-word registers.
//!
//! Workload: 4096 symbols per call, random points of the constellation with uniform noise of
//! ±0.05 per component, scale 1.
//!
//! The carrier loop `costas` with the constellation `qam16` (M44 §3) stays with the fallback on
//! every backend, as the PSK loop (the phase of every symbol depends on the detector output of
//! the one before). Its fallback is counted against the loop of GNU Radio's
//! `constellation_receiver_cb` with GNU Radio's 16-QAM `constellation_rect`
//! (`benches/refs/gr-constellation.c`, D1210): the same decision-directed loop with the same
//! detector `arg(y conj d)` (`cargf`) against the grid decision, `sincosf` instead of the table
//! lookup, the sector decision with two divisions instead of the product by `sqrt(E) / 2`, and no
//! lock detector. GNU Radio runs the loop bandwidth `theta = loop_bw / (damping + 1 / (4
//! damping))` of our loop constants and the frequency limits `+-2 pi max_offset`, so both loops
//! have the same gains (GNU Radio's in radians, ours in cycles). The workload is 16-QAM with one
//! sample per symbol, a start phase of 0.3 rad, a carrier offset of 0.0005 cycles per symbol and
//! uniform noise of ±0.05 per component, with the block defaults (`loop_bw` 0.02, `damping`
//! 0.7071, `max_offset` 0.05).
//!
//! The equalizer `equalize` with the constellation `qam16` runs the fallback's loop on every
//! backend (D841). Its fallback is counted against GNU Radio's `linear_equalizer` with
//! `adaptive_algorithm_nlms` deciding on the same `constellation_rect` (gr-constellation.c, the
//! loop of gr-equalizer.c of `equalize_fallback`, D838, D1210). The workload: the block default (11
//! taps, no feedback, `mu_cma` 0.05, `mu_dd` 0.02, `cma_symbols` 1000, `phase_bw` 0.02) with the
//! constellation `qam16` on 16-QAM through the channel `1 + 0.2 e^(j0.6) z^-1 + 0.05 e^(-j2)
//! z^-2` with a constant carrier phase of 0.2 rad and uniform noise of ±0.02 per component. The
//! state lives across the runs: the warm-up call covers the 1000 symbols in CMA mode, so the
//! counted calls run in decision mode, the steady state, as GNU Radio's decision-directed loop
//! with the same normalized step 0.02 from the centre-spike taps.

use std::hint::black_box;

use cref_sys::{GrConstellationReceiverCb, GrLinearEqualizerRect, GrRect};
use sdrk::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk::kernels::equalize::{EqConstellation, EqParams, Equalize, EqualizeState};
use sdrk::{C32, Fallback, Rate, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, FrameDecodeBlock, FrameDecodeConfig, FrameRotations, Packet, PacketFlags,
    QamDemodBlock, QamDemodConfig, QamModBlock, QamModConfig, SliceBlock, SliceConfig,
    SliceConstellation,
};
use sdrk_proto::frame::{self, Whitening};
use sdrk_core::Reference;
use sdrk_core::kernels::constellation::{Qam, QamSoftBits};

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32_pairs, c32_pairs_ref, c32s, snr_db};

/// The kernel `qam_soft_bits` over `n` symbols per call: `ours` through the facade, `fallback`
/// the fallback.
pub(crate) struct QamSoftBitsK {
    qam: Qam,
    input: Aligned<C32>,
    out: Aligned<f32>,
}

impl QamSoftBitsK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam) -> Self {
        let mut input = Aligned::<C32>::skewed(layout.n, layout.skew(8));
        for x in input.iter_mut() {
            let p = qam.point(rng.next_u32());
            *x = C32::new(p.re + 0.05 * rng.unit(), p.im + 0.05 * rng.unit());
        }
        QamSoftBitsK {
            qam,
            input,
            out: Aligned::skewed(layout.n * qam.bits_per_symbol(), layout.skew(4)),
        }
    }
}

/// One call with the scale 1 hidden from the optimizer (`black_box`), so that the fallback,
/// inlined here, keeps its products with the scale as every caller with a run-time scale does
/// (the block `slice` and `qam_demod`).
#[inline(never)]
fn soft_bits_call<K: QamSoftBits<C32>>(k: &K, qam: Qam, x: &[C32], out: &mut [f32]) {
    let _ = black_box(k.qam_soft_bits(qam, black_box(x), out, black_box(1.0)));
}

impl Kernel for QamSoftBitsK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => soft_bits_call(&Sdrk, self.qam, &self.input, &mut self.out),
                Impl::Fallback => {
                    soft_bits_call(&Fallback, self.qam, &self.input, &mut self.out);
                }
                other => unreachable!("qam_soft_bits has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The soft bits against the definition of M44 §2 computed in `f64`.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.run(which, 1);
        let mut want = Vec::with_capacity(self.out.len());
        for y in self.input.iter() {
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

// ---------------------------------------------------------------------------------------------
// GNU Radio's 16-QAM constellation_rect

/// The 16-QAM points of ours in symbol order, as pairs.
fn qam16_points() -> Vec<[f32; 2]> {
    Qam::Qam16
        .point_table()
        .iter()
        .map(|p| [p.re, p.im])
        .collect()
}

/// GNU Radio's `constellation_rect` of our unit-power 16-QAM: 4 by 4 sectors of the width of
/// the level spacing `2 / sqrt(10)`, no normalization.
fn gr_rect(points: &[[f32; 2]]) -> GrRect<'_> {
    #[allow(clippy::cast_possible_truncation, reason = "the f32 width")]
    let width = (2.0 / 10f64.sqrt()) as f32;
    GrRect {
        points,
        real_sectors: 4,
        imag_sectors: 4,
        width_real: width,
        width_imag: width,
    }
}

/// The sector decision of GNU Radio's `constellation_rect` in f64: the point nearest to the
/// sector centre of `int(x / width + 2)` per axis, clamped to `0 ..= 3`; for our grid that is
/// the level of the sector.
fn gr_decide_f64(x: f64) -> f64 {
    let width = 2.0 / 10f64.sqrt();
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the truncation of the C code"
    )]
    let s = ((x / width + 2.0) as i64).clamp(0, 3);
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let level = (2 * s - 3) as f64;
    level / 10f64.sqrt()
}

/// 16-QAM symbols (random points) through the symbol-spaced `channel` (`(delay, magnitude,
/// phase)`), turned by `phase0 + 2 pi offset k`, plus uniform noise of `noise` per component.
fn qam16_input(
    rng: &mut Pcg32,
    layout: Layout,
    channel: &[(usize, f64, f64)],
    phase0: f64,
    offset: f64,
    noise: f64,
) -> Aligned<C32> {
    let n = layout.n;
    let skew = layout.skew(core::mem::size_of::<C32>());
    let symbols: Vec<C32> = (0..n).map(|_| Qam::Qam16.point(rng.next_u32())).collect();
    let mut input = Aligned::<C32>::skewed(n, skew);
    for (k, s) in input.iter_mut().enumerate() {
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for &(d, mag, ph) in channel {
            if let Some(x) = k.checked_sub(d).and_then(|j| symbols.get(j)) {
                let (gs, gc) = ph.sin_cos();
                let (xr, xi) = (f64::from(x.re), f64::from(x.im));
                re += mag * (xr * gc - xi * gs);
                im += mag * (xr * gs + xi * gc);
            }
        }
        #[allow(clippy::cast_precision_loss, reason = "a symbol index")]
        let turns = (phase0 / core::f64::consts::TAU + offset * k as f64).fract();
        let (sp, cp) = (core::f64::consts::TAU * turns).sin_cos();
        let nr = f64::from(rng.unit()) * noise;
        let ni = f64::from(rng.unit()) * noise;
        #[allow(clippy::cast_possible_truncation, reason = "the f32 input")]
        {
            *s = C32::new(
                (re * cp - im * sp + nr) as f32,
                (re * sp + im * cp + ni) as f32,
            );
        }
    }
    input
}

// ---------------------------------------------------------------------------------------------
// costas qam16

/// The block defaults of `costas` (D604) with the constellation `qam16`.
const LOOP_BW: f64 = 0.02;
#[allow(clippy::approx_constant)]
const DAMPING: f64 = 0.7071;
const MAX_OFFSET: f64 = 0.05;
/// The start phase (rad), the carrier offset (cycles per symbol) and the noise per component.
const CR_PHASE0: f64 = 0.3;
const CR_OFFSET: f64 = 0.0005;
const CR_NOISE: f64 = 0.05;

fn costas_state() -> CostasState {
    CostasState::new(CostasConstellation::Qam16, LOOP_BW, DAMPING, MAX_OFFSET).expect("costas")
}

/// GNU Radio's `constellation_receiver_cb` with the loop bandwidth of our loop constants and the
/// frequency limits of our `max_offset`.
fn gr_receiver(points: &[[f32; 2]]) -> Option<GrConstellationReceiverCb> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the f32 parameters of the block"
    )]
    let (bw, f) = (
        (LOOP_BW / (DAMPING + 1.0 / (4.0 * DAMPING))) as f32,
        (core::f64::consts::TAU * MAX_OFFSET) as f32,
    );
    GrConstellationReceiverCb::new(gr_rect(points), bw, -f, f)
}

/// The f64 model of the `constellation_receiver_cb` loop (exact sine and cosine, the sector
/// decision, `-arg(sample conj(d))`), the accuracy reference of [`Impl::GnuRadio`].
fn gr_receiver_model(input: &[C32], alpha: f64, beta: f64) -> Vec<C32> {
    let fmax = core::f64::consts::TAU * MAX_OFFSET;
    let (mut phase, mut freq) = (0.0f64, 0.0f64);
    let mut out = Vec::with_capacity(input.len());
    for x in input {
        let (s, c) = phase.sin_cos();
        let (xr, xi) = (f64::from(x.re), f64::from(x.im));
        let (yr, yi) = (xr * c - xi * s, xr * s + xi * c);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a model value rounded to f32"
        )]
        out.push(C32::new(yr as f32, yi as f32));
        let (dr, di) = (gr_decide_f64(yr), gr_decide_f64(yi));
        let e = -(yi * dr - yr * di).atan2(yr * dr + yi * di);
        freq += beta * e;
        phase += freq + alpha * e;
        let two_pi = core::f64::consts::TAU;
        while phase > two_pi {
            phase -= two_pi;
        }
        while phase < -two_pi {
            phase += two_pi;
        }
        freq = freq.clamp(-fmax, fmax);
    }
    out
}

/// `costas` with the constellation `qam16` over `n` symbols.
pub(crate) struct CostasQamK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    points: Vec<[f32; 2]>,
    /// The entry: `costas_qam16_fallback`.
    fallback_entry: bool,
}

impl CostasQamK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, fallback_entry: bool) -> Self {
        let skew = layout.skew(core::mem::size_of::<C32>());
        CostasQamK {
            input: qam16_input(
                rng,
                layout,
                &[(0, 1.0, 0.0)],
                CR_PHASE0,
                CR_OFFSET,
                CR_NOISE,
            ),
            out: Aligned::skewed(layout.n, skew),
            points: qam16_points(),
            fallback_entry,
        }
    }
}

#[inline(never)]
fn costas<B: Costas>(backend: &B, state: &mut CostasState, input: &[C32], out: &mut [C32]) {
    let _ = black_box(backend.costas(state, black_box(input), out));
}

impl Kernel for CostasQamK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut state = costas_state();
        let mut gr = if which == Impl::GnuRadio {
            gr_receiver(&self.points)
        } else {
            None
        };
        for _ in 0..k {
            match which {
                Impl::Ours => costas(&Sdrk, &mut state, &self.input, &mut self.out),
                Impl::Fallback => costas(&Fallback, &mut state, &self.input, &mut self.out),
                Impl::GnuRadio => gr
                    .as_mut()
                    .expect("gnuradio")
                    .work(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                other => unreachable!("costas_qam16 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = if which == Impl::GnuRadio {
            let (alpha, beta) = gr_receiver(&self.points).map_or((0.0, 0.0), |g| g.gains());
            gr_receiver_model(&self.input, f64::from(alpha), f64::from(beta))
        } else {
            let mut want = vec![C32::default(); self.input.len()];
            let _ = Reference.costas(&mut costas_state(), &self.input, &mut want);
            want
        };
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours => !self.fallback_entry,
            Impl::Fallback => true,
            Impl::GnuRadio => self.fallback_entry && gr_receiver(&self.points).is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// equalize qam16

/// The taps of the block default.
const EQ_TAPS: usize = 11;
/// The carrier phase (rad) and the uniform noise per component of the workload.
const EQ_PHASE: f64 = 0.2;
const EQ_NOISE: f64 = 0.02;
/// The channel: `(delay in symbols, magnitude, phase)`.
const EQ_CHANNEL: [(usize, f64, f64); 3] = [(0, 1.0, 0.0), (1, 0.2, 0.6), (2, 0.05, -2.0)];

type EqState = EqualizeState<[C32; EQ_TAPS], [C32; EQ_TAPS]>;

fn eq_params() -> EqParams {
    EqParams {
        constellation: EqConstellation::Qam16,
        ..EqParams::default()
    }
}

fn eq_state() -> EqState {
    EqualizeState::new(
        [C32::default(); EQ_TAPS],
        [C32::default(); EQ_TAPS],
        eq_params(),
    )
    .expect("equalize")
}

/// The centre-spike taps of GNU Radio's equalizer (natural order).
fn gr_taps() -> [[f32; 2]; EQ_TAPS] {
    let mut t = [[0.0f32; 2]; EQ_TAPS];
    t[EQ_TAPS / 2] = [1.0, 0.0];
    t
}

/// GNU Radio's `linear_equalizer` on the 16-QAM `constellation_rect` for `n` symbols per call
/// with the normalized decision-mode step of ours.
fn gr_equalizer(n: usize, points: &[[f32; 2]]) -> Option<GrLinearEqualizerRect> {
    GrLinearEqualizerRect::new(&gr_taps(), eq_params().mu_dd, n, gr_rect(points))
}

/// The f64 model of GNU Radio's loop (the accuracy reference of [`Impl::GnuRadio`]): the FIR of
/// the window (oldest first) with the reversed taps, the sector decision, `e = d - y`,
/// `taps[i] += (mu / sum |x|^2) conj(x[i]) e`.
fn gr_equalizer_model(input: &[C32]) -> Vec<C32> {
    let n = EQ_TAPS;
    let mu = f64::from(eq_params().mu_dd);
    let mut taps: Vec<(f64, f64)> = gr_taps()
        .iter()
        .rev()
        .map(|t| (f64::from(t[0]), f64::from(t[1])))
        .collect();
    let mut window = vec![(0.0f64, 0.0f64); n];
    let mut out = Vec::with_capacity(input.len());
    for x in input {
        window.rotate_left(1);
        window[n - 1] = (f64::from(x.re), f64::from(x.im));
        let (mut yr, mut yi) = (0.0, 0.0);
        for (&(xr, xi), &(tr, ti)) in window.iter().zip(&taps) {
            yr += xr * tr - xi * ti;
            yi += xr * ti + xi * tr;
        }
        #[allow(
            clippy::cast_possible_truncation,
            reason = "a model value rounded to f32"
        )]
        out.push(C32::new(yr as f32, yi as f32));
        let (dr, di) = (gr_decide_f64(yr), gr_decide_f64(yi));
        let (er, ei) = (dr - yr, di - yi);
        let p: f64 = window.iter().map(|&(xr, xi)| xr * xr + xi * xi).sum();
        let m = mu / p;
        for (&(xr, xi), t) in window.iter().zip(taps.iter_mut()) {
            t.0 += m * (xr * er + xi * ei);
            t.1 += m * (xr * ei - xi * er);
        }
    }
    out
}

#[inline(never)]
fn equalize<B: Equalize>(backend: &B, state: &mut EqState, input: &[C32], out: &mut [C32]) {
    let _ = black_box(backend.equalize(state, black_box(input), out));
}

/// `equalize` with the constellation `qam16` over `n` symbols (see the module documentation).
pub(crate) struct EqualizeQamK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    points: Vec<[f32; 2]>,
    /// The state of the counted calls, carried across the runs.
    state: EqState,
    /// GNU Radio's equalizer of the counted calls (the fallback entry), carried across the runs.
    gr: Option<GrLinearEqualizerRect>,
    /// The entry: `equalize_qam16_fallback`.
    fallback_entry: bool,
}

impl EqualizeQamK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, fallback_entry: bool) -> Self {
        let skew = layout.skew(core::mem::size_of::<C32>());
        let points = qam16_points();
        let gr = if fallback_entry {
            gr_equalizer(layout.n, &points)
        } else {
            None
        };
        EqualizeQamK {
            input: qam16_input(rng, layout, &EQ_CHANNEL, EQ_PHASE, 0.0, EQ_NOISE),
            out: Aligned::skewed(layout.n, skew),
            points,
            state: eq_state(),
            gr,
            fallback_entry,
        }
    }
}

impl Kernel for EqualizeQamK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => equalize(&Sdrk, &mut self.state, &self.input, &mut self.out),
                Impl::Fallback => {
                    equalize(&Fallback, &mut self.state, &self.input, &mut self.out);
                }
                Impl::GnuRadio => self
                    .gr
                    .as_mut()
                    .expect("gnuradio")
                    .work(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                other => unreachable!("equalize_qam16 has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The SNR of one call from the start state against the float32 model of `Reference` (the
    /// kernel is bit-exact, so every implementation gives the cap of [`snr_db`]); GNU Radio's
    /// loop against its f64 model.
    fn accuracy(&mut self, which: Impl) -> f64 {
        let mut got = vec![C32::default(); self.input.len()];
        let mut state = eq_state();
        match which {
            Impl::Ours => equalize(&Sdrk, &mut state, &self.input, &mut got),
            Impl::Fallback => equalize(&Fallback, &mut state, &self.input, &mut got),
            Impl::GnuRadio => {
                if let Some(mut gr) = gr_equalizer(self.input.len(), &self.points) {
                    gr.work(c32_pairs_ref(&self.input), c32_pairs(&mut got));
                }
                return snr_db(&c32s(&gr_equalizer_model(&self.input)), &c32s(&got));
            }
            other => unreachable!("equalize_qam16 has no {other:?}"),
        }
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.equalize(&mut eq_state(), &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours => !self.fallback_entry,
            Impl::Fallback => true,
            Impl::GnuRadio => self.fallback_entry && self.gr.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}

/// The samples per symbol of the block entries `qam_mod_block` and `qam_demod_block`: the
/// 16-QAM of the flows of M44 §5 (25 kBd at 200 kHz).
const BLOCK_SPS: usize = 8;

fn block_rate(hz: u64) -> Rate {
    Rate::hz(hz).expect("rate")
}

fn leaked_arena(bytes: usize) -> ArenaStorage<'static> {
    // The block lives as long as the measurement binary; its arena is leaked once.
    let mem: &'static mut [u8] = Box::leak(vec![0u8; bytes + 16].into_boxed_slice());
    ArenaStorage::new(mem)
}

fn qam_mod_config() -> QamModConfig {
    QamModConfig {
        order: 16,
        samples_per_symbol: BLOCK_SPS,
        ..QamModConfig::default()
    }
}

fn qam_mod_block() -> QamModBlock<ArenaStorage<'static>> {
    let cfg = qam_mod_config();
    let rate = block_rate(100_000);
    let bytes = QamModBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    QamModBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("qam_mod block")
}

/// Random soft bits `+-1` of `n` 16-QAM symbols.
fn qam16_bits(rng: &mut Pcg32, n: usize) -> Vec<f32> {
    (0..4 * n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The block `qam_mod` (16-QAM, 8 samples per symbol, roll-off 0.35, span 10) writing `n` output
/// samples (`n / 8` symbols of random bits) per call.
pub(crate) struct QamModBlockK {
    block: QamModBlock<ArenaStorage<'static>>,
    bits: Vec<f32>,
    out: Vec<C32>,
}

impl QamModBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        QamModBlockK {
            block: qam_mod_block(),
            bits: qam16_bits(rng, layout.n / BLOCK_SPS),
            out: vec![C32::default(); layout.n],
        }
    }
}

#[inline(never)]
fn qam_mod_call(block: &mut QamModBlock<ArenaStorage<'static>>, bits: &[f32], out: &mut [C32]) {
    let _ = block.process(black_box(bits), out);
}

/// `qam_mod` in `f64` from a zero history: the points of M44 §1 (`(2 i - 3) / sqrt(10)` per axis
/// from the Gray code) at every `sps`-th sample, filtered with `rrc(0.35, sps, 10) sqrt(sps)`.
fn qam_mod_f64(bits: &[f32]) -> Vec<(f64, f64)> {
    let cfg = qam_mod_config();
    let sps = cfg.samples_per_symbol;
    let mut taps = vec![0.0f64; sdrk_blocks::design::rrc_len(sps, cfg.span).expect("rrc")];
    sdrk_blocks::design::rrc(cfg.rolloff, sps, cfg.span, &mut taps).expect("rrc");
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let gain = (sps as f64).sqrt();
    let level = |b: &[f32]| {
        let g = usize::from(b[0] > 0.0) * 2 + usize::from(b[1] > 0.0);
        let i = (0..4usize).find(|&i| i ^ (i >> 1) == g).unwrap_or(0);
        #[allow(clippy::cast_precision_loss, reason = "a small integer")]
        let v = (2 * i) as f64 - 3.0;
        v / 10f64.sqrt()
    };
    let syms: Vec<(f64, f64)> = bits
        .chunks_exact(4)
        .map(|b| (level(&b[..2]), level(&b[2..])))
        .collect();
    let n = syms.len() * sps;
    (0..n)
        .map(|t| {
            let (mut re, mut im) = (0.0, 0.0);
            for (k, h) in taps.iter().enumerate() {
                if let Some(j) = t.checked_sub(k).filter(|j| j % sps == 0)
                    && let Some(&(sr, si)) = syms.get(j / sps)
                {
                    re += h * gain * sr;
                    im += h * gain * si;
                }
            }
            (re, im)
        })
        .collect()
}

impl Kernel for QamModBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => qam_mod_call(&mut self.block, &self.bits, &mut self.out),
                other => unreachable!("qam_mod_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// One call from the reset state against the `f64` model.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        qam_mod_call(&mut self.block, &self.bits, &mut self.out);
        let want: Vec<f64> = qam_mod_f64(&self.bits)
            .into_iter()
            .flat_map(|(re, im)| [re, im])
            .collect();
        snr_db(&want, &c32s(&self.out))
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

fn qam_demod_block() -> QamDemodBlock<ArenaStorage<'static>> {
    let cfg = QamDemodConfig {
        order: 16,
        symbol_rate_hz: 25_000.0,
        ..QamDemodConfig::default()
    };
    let rate = block_rate(200_000);
    let bytes =
        QamDemodBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    QamDemodBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("qam_demod block")
}

/// Calls of the warm-up of `qam_demod_block`: 6 calls of 512 symbols cover the 1000 CMA symbols
/// of the equalizer and bring the loops to lock, so the counted calls run in the steady state.
const DEMOD_WARMUP: usize = 6;

/// The block `qam_demod` (16-QAM at 25 kBd and 200 kHz, the block defaults: 11 equalizer taps,
/// `carrier_bw` 0.005, `sync_bw` 0.005) reading `n` input samples per call: the output of
/// `qam_mod` for `n / 8` random symbols (repeated every call; `n / 8` is whole symbols, so the
/// symbol clock stays on the grid), turned by 0.3 rad with uniform noise of ±0.02 per component.
/// The state lives across the calls; the warm-up runs in `new`.
pub(crate) struct QamDemodBlockK {
    block: QamDemodBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    symbols: Vec<u32>,
    out: Vec<f32>,
    produced: usize,
}

#[inline(never)]
fn qam_demod_call(
    block: &mut QamDemodBlock<ArenaStorage<'static>>,
    input: &[C32],
    out: &mut [f32],
) -> usize {
    block.process(black_box(input), out).map_or(0, |p| p.produced)
}

impl QamDemodBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let bits = qam16_bits(rng, n / BLOCK_SPS);
        let symbols = bits
            .chunks_exact(4)
            .map(|b| b.iter().fold(0u32, |g, &x| (g << 1) | u32::from(x > 0.0)))
            .collect();
        let mut tx = qam_mod_block();
        let mut sig = vec![C32::default(); n];
        // Two passes: the second starts with the pulse tails of the first, as the repeated
        // input does at every call.
        let _ = tx.process(&bits, &mut sig);
        let _ = tx.process(&bits, &mut sig);
        let (s, c) = 0.3f32.sin_cos();
        let input = sig
            .iter()
            .map(|x| {
                C32::new(
                    x.re * c - x.im * s + 0.02 * rng.unit(),
                    x.re * s + x.im * c + 0.02 * rng.unit(),
                )
            })
            .collect();
        let mut k = QamDemodBlockK {
            block: qam_demod_block(),
            input,
            symbols,
            // Room for every soft bit of a call (a call may give a symbol more than nominal).
            out: vec![0.0; n / BLOCK_SPS * 4 + 64],
            produced: 0,
        };
        for _ in 0..DEMOD_WARMUP {
            k.produced = qam_demod_call(&mut k.block, &k.input, &mut k.out);
        }
        k
    }
}

impl Kernel for QamDemodBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    self.produced = qam_demod_call(&mut self.block, &self.input, &mut self.out);
                }
                other => unreachable!("qam_demod_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when the symbols of one call after the warm-up equal the sent ones at one of the four
    /// rotations and some lag (the input repeats, so the lag is taken modulo the call), 0
    /// otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.run(which, 1);
        let got: Vec<u32> = self.out[..self.produced]
            .chunks_exact(4)
            .map(|b| b.iter().fold(0u32, |g, &x| (g << 1) | u32::from(x > 0.0)))
            .collect();
        let m = self.symbols.len();
        let ok = got.len() + 2 >= m
            && (0..4u32).any(|r| {
                (0..m).any(|lag| {
                    got.iter().enumerate().all(|(j, &s)| {
                        let back = sdrk_core::kernels::constellation::rotate_symbol(s, 2, 4 - r);
                        back == self.symbols[(j + m - lag) % m]
                    })
                })
            });
        if ok { 999.0 } else { 0.0 }
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

/// The block `frame_decode` with `rotations` `four` and 4 bits per symbol over `n` soft bits:
/// frames of a 4-byte preamble and a random 32-byte payload back to back (the workload of
/// `frame_decode_block`), every symbol of 4 bits turned by one quarter turn.
pub(crate) struct FrameDecodeFourBlockK {
    block: FrameDecodeBlock<ArenaStorage<'static>>,
    soft: Vec<f32>,
    payloads: Vec<Vec<u8>>,
    got: Vec<Packet>,
    produced: usize,
}

/// Payload bytes and preamble bytes of the frames of `frame_decode_four_block`.
const FOUR_PAYLOAD: usize = 32;
const FOUR_PREAMBLE: usize = 4;

impl FrameDecodeFourBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let per = frame::frame_bits(FOUR_PAYLOAD, FOUR_PREAMBLE);
        let mut bits: Vec<u8> = Vec::with_capacity(n);
        let mut payloads = Vec::new();
        while bits.len() + per <= n {
            let payload: Vec<u8> = (0..FOUR_PAYLOAD).map(|_| rng.next_u32() as u8).collect();
            let p = Packet::new(0, PacketFlags::NONE, &payload).expect("packet");
            let mut f = vec![0u8; per];
            frame::encode(&p, FOUR_PREAMBLE, Whitening::None, &mut f).expect("frame");
            bits.extend(f);
            payloads.push(payload);
        }
        bits.resize(n, 0);
        let soft = bits
            .chunks_exact(4)
            .flat_map(|c| {
                let g = c.iter().fold(0u32, |g, &b| (g << 1) | u32::from(b));
                let t = frame::rotate_group(g, 2, 1);
                (0..4).rev().map(move |j| if (t >> j) & 1 == 1 { 1.0f32 } else { -1.0 })
            })
            .collect();
        let cfg = FrameDecodeConfig {
            rotations: FrameRotations::Four,
            bits_per_symbol: 4,
            ..FrameDecodeConfig::default()
        };
        let block = FrameDecodeBlock::new(&cfg, block_rate(100_000), &mut ArenaStorage::new(&mut []))
            .expect("frame_decode");
        FrameDecodeFourBlockK {
            block,
            soft,
            payloads,
            got: vec![Packet::default(); n / 56 + 1],
            produced: 0,
        }
    }
}

#[inline(never)]
fn decode_four(
    block: &mut FrameDecodeBlock<ArenaStorage<'static>>,
    soft: &[f32],
    got: &mut [Packet],
) -> usize {
    block.process(black_box(soft), got).map_or(0, |p| p.produced)
}

impl Kernel for FrameDecodeFourBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    self.produced = decode_four(&mut self.block, &self.soft, &mut self.got);
                }
                other => unreachable!("frame_decode_four_block has no {other:?}"),
            }
            black_box(&self.got);
        }
    }

    /// 999 when one call from the reset state gives every frame with `CRC_OK` and the rotation
    /// 1, 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        self.run(which, 1);
        let got = &self.got[..self.produced];
        let ok = got.len() == self.payloads.len()
            && got.iter().zip(&self.payloads).all(|(g, p)| {
                g.data() == p.as_slice()
                    && g.flags.contains(PacketFlags::CRC_OK)
                    && frame::rotation_of(g.flags) == 1
            });
        if ok { 999.0 } else { 0.0 }
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
