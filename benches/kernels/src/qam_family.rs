//! The QAM entries of M44: `qam_soft_bits_qam16`, `qam_soft_bits_qam64`,
//! `symbol_demap_qam16_block`, `symbol_demap_qam64_block`, `costas_qam16`, `costas_qam16_fallback`,
//! `costas_qam64`, `costas_qam64_fallback`, `equalize_qam16`, `equalize_qam16_fallback`,
//! `equalize_qam64`, `equalize_qam64_fallback`, `qam_mod_block`, `qam_demod_block` and
//! `frame_decode_four_block`, the entries of M44.1 `psk_mod_block`, `symbol_map_block`,
//! `iq_offset_block`, `qam_acquire_block`, `qam_frame_encode_block` and `qam_frame_decode_block`,
//! the entries of M44.1.1 `pulse_shape_block` and `pulse_match_block`, and the entries of M44.2
//! `qam_soft_bits_qpsk`, `qam_soft_bits_qam256`, `qam_soft_bits_qam1024`, `costas_qam256`,
//! `costas_qam1024`, `equalize_qam256` and `equalize_qam1024` (D1507), and
//! `symbol_demap_qam256_block`, `costas_qam256_fallback`, `equalize_qam256_fallback`,
//! `qam_mod_qam256_block` and `qam_demod_qam256_block` (D1524), and `costas_qam1024_fallback` and
//! `equalize_qam1024_fallback` (D1545).
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `qam_soft_bits_qam16` | the kernel `qam_soft_bits` with `Qam16` through the facade | our fallback (`info`) | one input symbol (four soft bits) |
//! | `qam_soft_bits_qam64` | the kernel `qam_soft_bits` with `Qam64` through the facade | our fallback (`info`) | one input symbol (six soft bits) |
//! | `symbol_demap_qam16_block` | `sdrk_blocks::SymbolDemapBlock` with `qam16` | none (`info`) | one input symbol (four soft bits) |
//! | `symbol_demap_qam64_block` | `sdrk_blocks::SymbolDemapBlock` with `qam64` | none (`info`) | one input symbol (six soft bits) |
//! | `costas_qam16` | facade (= the fallback) | none (`info`) | one symbol |
//! | `costas_qam16_fallback` | our fallback | GNU Radio's `constellation_receiver_cb` | one symbol |
//! | `equalize_qam16` | facade (= the fallback's loop, D841) | none (`info`) | one symbol |
//! | `equalize_qam16_fallback` | our fallback | GNU Radio's `linear_equalizer` on a `constellation_rect` | one symbol |
//! | `costas_qam64` | facade (= the fallback), 64-QAM | none (`info`) | one symbol |
//! | `costas_qam64_fallback` | our fallback, 64-QAM | GNU Radio's `constellation_receiver_cb` on the 64-QAM `constellation_rect` | one symbol |
//! | `equalize_qam64` | facade (= the fallback's loop, D841), 64-QAM | none (`info`) | one symbol |
//! | `equalize_qam64_fallback` | our fallback, 64-QAM | GNU Radio's `linear_equalizer` on the 64-QAM `constellation_rect` | one symbol |
//! | `qam_mod_block` | `sdrk_blocks::QamModBlock`, 16-QAM | none (`info`) | one output sample |
//! | `psk_mod_block` | `sdrk_blocks::PskModBlock`, QPSK | none (`info`) | one output sample |
//! | `qam_demod_block` | `sdrk_blocks::QamDemodBlock`, 16-QAM | none (`info`) | one input sample |
//! | `frame_decode_four_block` | `sdrk_blocks::FrameDecodeBlock` with `rotations` `four` | none (`info`) | one input sample (one soft bit) |
//! | `symbol_map_block` | `sdrk_blocks::SymbolMapBlock` with `qam16` | none (`info`) | one output symbol |
//! | `iq_offset_block` | `sdrk_blocks::IqOffsetBlock`, `imag`, 8 samples per symbol (a delay of 4 samples) | none (`info`) | one sample |
//! | `qam_acquire_block` | `sdrk_blocks::QamAcquireBlock`, `correct` false, searching | none (`info`) | one input symbol |
//! | `qam_frame_encode_block` | `sdrk_blocks::QamFrameEncodeBlock`, 16-QAM | none (`info`) | one output sample |
//! | `qam_frame_decode_block` | `sdrk_blocks::QamFrameDecodeBlock`, 16-QAM | none (`info`) | one input sample |
//! | `pulse_shape_block` | `sdrk_blocks::PulseShapeBlock`, 8 samples per symbol, roll-off 0.35, span 10 | none (`info`) | one output sample |
//! | `pulse_match_block` | `sdrk_blocks::PulseMatchBlock`, 25 kBd at 200 kHz, roll-off 0.35, span 10 | none (`info`) | one input sample |
//! | `qam_soft_bits_qpsk`, `qam_soft_bits_qam256`, `qam_soft_bits_qam1024` | the kernel `qam_soft_bits` with `Qpsk`, `Qam256`, `Qam1024` through the facade | our fallback (`info`) | one input symbol (2, 8 or 10 soft bits) |
//! | `costas_qam256`, `costas_qam1024` | facade (= the fallback), the noise and the start phase scaled by the level spacing | none (`info`) | one symbol |
//! | `equalize_qam256`, `equalize_qam1024` | facade (= the fallback's loop), the noise scaled by the level spacing | none (`info`) | one symbol |
//! | `symbol_demap_qam256_block` | `sdrk_blocks::SymbolDemapBlock` with `qam256` | none (`info`) | one input symbol (eight soft bits) |
//! | `costas_qam256_fallback` | our fallback, 256-QAM | GNU Radio's `constellation_receiver_cb` on the 256-QAM `constellation_rect` | one symbol |
//! | `equalize_qam256_fallback` | our fallback, 256-QAM | GNU Radio's `linear_equalizer` on the 256-QAM `constellation_rect` | one symbol |
//! | `costas_qam1024_fallback` | our fallback, 1024-QAM | GNU Radio's `constellation_receiver_cb` on the 1024-QAM `constellation_rect` | one symbol |
//! | `equalize_qam1024_fallback` | our fallback, 1024-QAM | GNU Radio's `linear_equalizer` on the 1024-QAM `constellation_rect` | one symbol |
//! | `qam_mod_qam256_block` | `sdrk_blocks::QamModBlock`, 256-QAM | none (`info`) | one output sample |
//! | `qam_demod_qam256_block` | `sdrk_blocks::QamDemodBlock`, 256-QAM, the steady state | none (`info`) | one input sample |
//!
//! The kernel `qam_soft_bits` (I44-1, D1412) puts both components of every symbol on the
//! integer grid (`v = x √E`) and forms the piecewise-linear max-log soft bits of M44 §2 (`v`,
//! `2 - |v|`, and for 64-QAM `4 - |v|`, `2 - ||v| - 4|`); the SIMD backend has a hand-written
//! path per level (D1414), the fallback is the component loop that the compiler vectorizes
//! where it can. No library computes the same soft bits: liquid-dsp's `modemcf_demodulate_soft`
//! and GNU Radio's `calc_soft_dec` search the points for the exact max-log or log-sum value
//! (D1201), so the kernel entries are our own code, judged against our fallback as information
//! (`Family::Own`). The block `symbol_demap` calls the kernel through the facade; its entries
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
//! `psk_mod_block` (M44.1 §8, the baseline before `psk_mod` becomes a composite) is the block
//! `psk_mod` with coherent QPSK at 8 samples per symbol, roll-off 0.35 and span 10 (the flow
//! `qpsk_transmitter`), writing 4096 output samples (512 symbols) per call, as `qam_mod_block`.
//! The building blocks and composites of M44.1 (§8) have entries of their own, information like
//! the blocks above: `symbol_map_block` (`qam16`, 4096 symbols per call, the first stage of
//! `qam_mod`), `iq_offset_block` (`imag`, 4 samples, the last stage of `psk_mod` with `offset`),
//! `qam_acquire_block` (`correct` false while it searches on noise, the share of the acquisition
//! in `qam_demod_search_block`), `qam_frame_encode_block` (`frame_encode` -> `qam_mod`, 16-QAM at
//! 8 samples per symbol, 32-byte frames back to back) and `qam_frame_decode_block` (`qam_demod`
//! with the workload of `qam_demod_block` carrying 32-byte frames -> `frame_decode` with
//! `rotations` `four` and the frame watch). The pulse filters of M44.1.1 (§3) have entries of
//! their own, information as well: `pulse_shape_block` (the second stage of `qam_mod`, the pulse
//! of `qam_mod_block` on its 16-QAM points, 4096 output samples per call) and `pulse_match_block`
//! (the first stage of `qam_demod`, the matched filter of `qam_demod_block` on its input, 4096
//! input samples per call); each is the kernel of its own entry (`fir_interpolate_short`, `fir`)
//! with the block around it.
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
//! state lives across the runs: the warm-up call covers the 1000 blind symbols (the multi-modulus
//! start of the square grids since M44.4, D2530; CMA before), so the
//! counted calls run in decision mode, the steady state, as GNU Radio's decision-directed loop
//! with the same normalized step 0.02 from the centre-spike taps.
//!
//! The 64-QAM entries `costas_qam64`, `costas_qam64_fallback`, `equalize_qam64` and
//! `equalize_qam64_fallback` (I44-11, D1601) are the same workloads with the constellation
//! `qam64` and 64-QAM points, against GNU Radio's `constellation_rect` with 8 by 8 sectors of the
//! width `2 / sqrt(42)` (the same C translations); `costas` starts at the phase 0.05 rad instead
//! of 0.3, at which the outer 64-QAM points are decided wrongly (D1425).
//!
//! The 256-QAM entries of M44.2 (D1524) are the same workloads with the constellation `qam256`:
//! `costas_qam256_fallback` and `equalize_qam256_fallback` against GNU Radio's
//! `constellation_rect` with 16 by 16 sectors of the width `2 / sqrt(170)`, the start phase
//! (0.025 rad) and the noise (±0.012 for `costas`, ±0.005 for `equalize`) scaled by the level
//! spacing as for `costas_qam256` and `equalize_qam256`. `symbol_demap_qam256_block` is the
//! block `symbol_demap` with `qam256` on the workload of `qam_soft_bits_qam256`.
//! `qam_mod_qam256_block` is `qam_mod_block` with `qam256` (8 bits per symbol, 512 symbols per
//! call). `qam_demod_qam256_block` is `qam_demod_block` with `qam256` and the block defaults, the
//! noise ±0.005 per component (the 16-QAM noise in units of the level spacing); its warm-up of 30
//! calls covers the blind start of 256-QAM (8000 multi-modulus symbols, D2509) and the first
//! check of the carrier loop (12288 symbols, at symbol 13056), so its counted calls measure the
//! steady state (`Track`, the equalizer in decision mode) rather than the mix of
//! `qam_demod_block`.
//!
//! The 1024-QAM references (D1545) are the workloads of `costas_qam1024` and `equalize_qam1024`
//! on our fallback: `costas_qam1024_fallback` and `equalize_qam1024_fallback` against GNU Radio's
//! `constellation_rect` with 32 by 32 sectors of the width `2 / sqrt(682)` (the C translations
//! take up to 1024 points and sectors, `SDRK_GR_MAX_POINTS`).

use std::hint::black_box;

use cref_sys::{
    GrChunksToSymbols, GrConstellationReceiverCb, GrFirBank, GrLinearEqualizerRect, GrPfbArb,
    GrRect, GrSoftDecoder, LiquidFirfiltCrcf, LiquidFirinterpCrcf, LiquidModemDemod, LiquidModemMap,
};

use super::psk_family::GrFront;
use sdrk::kernels::costas::{Costas, CostasConstellation, CostasState};
use sdrk::kernels::equalize::{EqConstellation, EqParams, Equalize, EqualizeState};
use sdrk::{C32, Fallback, Rate, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, FrameDecodeBlock, FrameDecodeConfig, FrameEncodeConfig, FrameRotations,
    IqOffsetBlock, IqOffsetConfig, IqPart, Packet, PacketFlags, PskModBlock, PskModConfig,
    PulseMatchBlock, PulseMatchConfig, PulseShapeBlock, QamAcquireBlock, QamAcquireConfig,
    QamDemodBlock, QamDemodConfig, QamFrameDecodeBlock, QamFrameDecodeConfig, QamFrameEncodeBlock,
    QamFrameEncodeConfig, QamModBlock, QamModConfig, Scrambler, SymbolDemapBlock,
    SymbolDemapConfig, SymbolDemapConstellation, SymbolMapBlock, SymbolMapConfig,
    SymbolMapConstellation,
};
use sdrk_core::Reference;
use sdrk_core::kernels::constellation::{Qam, QamSoftBits};
use sdrk_proto::frame::{self, Whitening};

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32_pairs, c32_pairs_ref, c32s, snr_db};

/// The reference of a soft-bit entry (issue #169, D2549): GNU Radio's `calc_soft_dec` or its
/// look-up table (`constellation_soft_decoder_cf`), or liquid-dsp's `modemcf_demodulate_soft`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SoftRef {
    /// No library (the entries of the kernel `qam_soft_bits` of the other constellations).
    None,
    GrCalc,
    GrLut,
    Liquid,
}

/// The noise power of GNU Radio's decoder (the default of its constellation) and the precision
/// of its look-up table (`2^8` by `2^8` cells).
const GR_NPWR: f32 = 1.0;
const GR_LUT_PRECISION: i32 = 8;

/// The library of a soft-bit entry with its buffers.
struct SoftRefs {
    mode: SoftRef,
    qam: Qam,
    gr: Option<GrSoftDecoder>,
    liquid: Option<LiquidModemDemod>,
    /// liquid's soft bytes and hard symbols of the last call, and its points.
    soft: Vec<u8>,
    syms: Vec<u32>,
    points: Vec<[f32; 2]>,
}

impl SoftRefs {
    fn new(qam: Qam, n: usize, mode: SoftRef) -> Self {
        let k = qam.bits_per_symbol();
        let gr = match mode {
            SoftRef::GrCalc => GrSoftDecoder::new(&qam_points(qam), k, GR_NPWR, 0),
            SoftRef::GrLut => GrSoftDecoder::new(&qam_points(qam), k, GR_NPWR, GR_LUT_PRECISION),
            _ => None,
        };
        let mut liquid = if mode == SoftRef::Liquid {
            LiquidModemDemod::new(1 << k)
        } else {
            None
        };
        let points = liquid
            .as_mut()
            .map(LiquidModemDemod::points)
            .unwrap_or_default();
        SoftRefs {
            mode,
            qam,
            gr,
            liquid,
            soft: vec![0; n * k],
            syms: vec![0; n],
            points,
        }
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::GnuRadio => self.gr.is_some(),
            Impl::Liquid => self.liquid.is_some(),
            _ => false,
        }
    }

    /// One call of the library on `x`; GNU Radio's soft values go to `out`, liquid's bytes to the
    /// buffers of the entry.
    fn run(&mut self, which: Impl, x: &[C32], out: &mut [f32]) {
        match which {
            Impl::GnuRadio => self
                .gr
                .as_mut()
                .expect("gnuradio")
                .work(black_box(c32_pairs_ref(x)), out),
            Impl::Liquid => self.liquid.as_mut().expect("liquid").demod_soft(
                black_box(c32_pairs_ref(x)),
                &mut self.soft,
                &mut self.syms,
            ),
            other => unreachable!("a soft-bit entry has no {other:?} as a library"),
        }
    }

    /// GNU Radio: the SNR against the exact log-likelihood ratio of `calc_soft_dec` in `f64`
    /// (the table method is quantized, so its SNR is lower); liquid: 999 when its hard symbol is
    /// the point of its own constellation nearest to the input and every soft byte is on the side
    /// of the bit of that symbol (127 is zero), 0 otherwise (liquid's labelling and scale are its
    /// own, so the values are not comparable with ours).
    fn accuracy(&mut self, which: Impl, x: &[C32], got: &[f32]) -> f64 {
        let k = self.qam.bits_per_symbol();
        if which == Impl::GnuRadio {
            let pts: Vec<(f64, f64)> = qam_points(self.qam)
                .iter()
                .map(|p| (f64::from(p[0]), f64::from(p[1])))
                .collect();
            let np = f64::from(GR_NPWR);
            let mut want = Vec::with_capacity(got.len());
            for y in x {
                let d: Vec<f64> = pts
                    .iter()
                    .map(|p| {
                        let (dr, di) = (f64::from(y.re) - p.0, f64::from(y.im) - p.1);
                        -(dr * dr + di * di) / np
                    })
                    .collect();
                let top = d.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for j in 0..k {
                    let (mut one, mut zero) = (0.0f64, 0.0f64);
                    for (i, &a) in d.iter().enumerate() {
                        if (i >> (k - 1 - j)) & 1 == 1 {
                            one += (a - top).exp();
                        } else {
                            zero += (a - top).exp();
                        }
                    }
                    want.push(one.ln() - zero.ln());
                }
            }
            let got: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
            return snr_db(&want, &got);
        }
        let ok = x.iter().enumerate().all(|(i, y)| {
            let s = self.syms[i] as usize;
            let nearest = self
                .points
                .iter()
                .enumerate()
                .map(|(j, p)| {
                    let (dr, di) = (y.re - p[0], y.im - p[1]);
                    (dr * dr + di * di, j)
                })
                .fold(
                    (f32::INFINITY, 0usize),
                    |a, b| if b.0 < a.0 { b } else { a },
                )
                .1;
            nearest == s
                && (0..k).all(|j| (self.soft[i * k + j] > 127) == ((s >> (k - 1 - j)) & 1 == 1))
        });
        if ok { 999.0 } else { 0.0 }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.mode) {
            (Impl::GnuRadio, SoftRef::GrLut) => {
                "GNU Radio constellation_soft_decoder_cf with the look-up table of gen_soft_dec_lut (precision 8, npwr 1; constellation.cc of gr-digital; exact log-likelihood ratio, not max-log; the table lookup and a copy of 2 to 3 floats per symbol, no vector allocation)"
            }
            (Impl::GnuRadio, _) => {
                "GNU Radio constellation_soft_decoder_cf with calc_soft_dec (constellation.cc of gr-digital; the exact log-likelihood ratio over all points, npwr 1, not max-log; no vector allocation per symbol, so faster than GNU Radio's object code)"
            }
            (Impl::Liquid, _) => {
                "liquid-dsp 1.8.3 modemcf_demodulate_soft of its own QAM modem (modem_common.proto.c; soft bytes with liquid's scale and labelling; the hard symbol and the soft bits per symbol)"
            }
            _ => "sdrk",
        }
    }
}

/// The kernel `qam_soft_bits` over `n` symbols per call: `ours` through the facade, `fallback`
/// the fallback.
pub(crate) struct QamSoftBitsK {
    qam: Qam,
    input: Aligned<C32>,
    out: Aligned<f32>,
    /// GNU Radio's decoder or liquid's demodulator of the entry (issue #169).
    refs: SoftRefs,
}

impl QamSoftBitsK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam, mode: SoftRef) -> Self {
        let mut input = Aligned::<C32>::skewed(layout.n, layout.skew(8));
        for x in input.iter_mut() {
            let p = qam.point(rng.next_u32());
            *x = C32::new(p.re + 0.05 * rng.unit(), p.im + 0.05 * rng.unit());
        }
        QamSoftBitsK {
            qam,
            input,
            out: Aligned::skewed(layout.n * qam.bits_per_symbol(), layout.skew(4)),
            refs: SoftRefs::new(qam, layout.n, mode),
        }
    }
}

/// One call with the scale 1 hidden from the optimizer (`black_box`), so that the fallback,
/// inlined here, keeps its products with the scale as every caller with a run-time scale does
/// (the block `symbol_demap` and `qam_demod`).
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
                Impl::GnuRadio | Impl::Liquid => self.refs.run(which, &self.input, &mut self.out),
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
        if matches!(which, Impl::GnuRadio | Impl::Liquid) {
            return self.refs.accuracy(which, &self.input, &self.out);
        }
        let mut want = Vec::with_capacity(self.out.len());
        for y in self.input.iter() {
            axis_f64(self.qam, y.re, &mut want);
            axis_f64(self.qam, y.im, &mut want);
        }
        let got: Vec<f64> = self.out.iter().map(|&v| f64::from(v)).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback) || self.refs.supports(which)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio | Impl::Liquid => self.refs.describe(which),
            _ => "sdrk",
        }
    }
}

/// The block `symbol_demap` with a QAM constellation over `n` symbols per call.
pub(crate) struct SymbolDemapQamBlockK {
    qam: Qam,
    block: SymbolDemapBlock<C32, ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<f32>,
    /// GNU Radio's decoder or liquid's demodulator of the entry (issue #169).
    refs: SoftRefs,
}

impl SymbolDemapQamBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam, mode: SoftRef) -> Self {
        let constellation = match qam {
            Qam::Qpsk => SymbolDemapConstellation::Qpsk,
            Qam::Qam16 => SymbolDemapConstellation::Qam16,
            Qam::Qam64 => SymbolDemapConstellation::Qam64,
            Qam::Qam256 => SymbolDemapConstellation::Qam256,
            Qam::Qam1024 => SymbolDemapConstellation::Qam1024,
            other => unreachable!("no symbol_demap constellation for {other:?}"),
        };
        let cfg = SymbolDemapConfig {
            constellation,
            scale: 1.0,
        };
        let block = SymbolDemapBlock::new(
            &cfg,
            Rate::hz(25_000).expect("rate"),
            &mut ArenaStorage::new(&mut []),
        )
        .expect("symbol_demap block");
        let input = (0..layout.n)
            .map(|_| {
                let p = qam.point(rng.next_u32());
                C32::new(p.re + 0.05 * rng.unit(), p.im + 0.05 * rng.unit())
            })
            .collect();
        SymbolDemapQamBlockK {
            qam,
            block,
            input,
            out: vec![0.0; layout.n * qam.bits_per_symbol()],
            refs: SoftRefs::new(qam, layout.n, mode),
        }
    }
}

#[inline(never)]
fn symbol_demap_call(
    block: &mut SymbolDemapBlock<C32, ArenaStorage<'static>>,
    x: &[C32],
    out: &mut [f32],
) {
    let _ = block.process(black_box(x), out);
}

/// The soft bits of one axis value in `f64` (M44.2 §2): `t_0 = x √E`, then
/// `t_i = 2^(b - i) - |t_(i - 1)|` for the `b` bits per axis.
fn axis_f64(qam: Qam, x: f32, out: &mut Vec<f64>) {
    let b = qam.bits_per_axis();
    let mut t = f64::from(x) * qam.energy().sqrt();
    out.push(t);
    for i in 1..b {
        t = f64::from(1u32 << (b - i)) - t.abs();
        out.push(t);
    }
}

impl Kernel for SymbolDemapQamBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    symbol_demap_call(&mut self.block, &self.input, &mut self.out);
                }
                Impl::GnuRadio | Impl::Liquid => self.refs.run(which, &self.input, &mut self.out),
                other => unreachable!("symbol_demap_qam has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The soft bits against the definition of M44 §2 computed in `f64`.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        if matches!(which, Impl::GnuRadio | Impl::Liquid) {
            self.run(which, 1);
            return self.refs.accuracy(which, &self.input, &self.out);
        }
        symbol_demap_call(&mut self.block, &self.input, &mut self.out);
        let mut want = Vec::with_capacity(self.out.len());
        for y in &self.input {
            axis_f64(self.qam, y.re, &mut want);
            axis_f64(self.qam, y.im, &mut want);
        }
        let got: Vec<f64> = self.out.iter().map(|&v| f64::from(v)).collect();
        snr_db(&want, &got)
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback) || self.refs.supports(which)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio | Impl::Liquid => self.refs.describe(which),
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GNU Radio's 16-QAM, 64-QAM and 256-QAM constellation_rect

/// The points of `qam` of ours in symbol order, as pairs.
fn qam_points(qam: Qam) -> Vec<[f32; 2]> {
    qam.point_table().iter().map(|p| [p.re, p.im]).collect()
}

/// GNU Radio's `constellation_rect` of our unit-power `qam`: `m` by `m` sectors (`m` the levels
/// per axis) of the width of the level spacing `2 / sqrt(E)` (16-QAM `2 / sqrt(10)`, 64-QAM
/// `2 / sqrt(42)`, 256-QAM `2 / sqrt(170)`), no normalization.
fn gr_rect(qam: Qam, points: &[[f32; 2]]) -> GrRect<'_> {
    #[allow(clippy::cast_possible_truncation, reason = "the f32 width")]
    let width = (2.0 / qam.energy().sqrt()) as f32;
    #[allow(clippy::cast_possible_truncation, reason = "at most 32 levels")]
    let m = qam.levels_per_axis() as u32;
    GrRect {
        points,
        real_sectors: m,
        imag_sectors: m,
        width_real: width,
        width_imag: width,
    }
}

/// The sector decision of GNU Radio's `constellation_rect` of `qam` in f64: the point nearest
/// to the sector centre of `int(x / width + m / 2)` per axis, clamped to `0 ..= m - 1`; for our
/// grid that is the level of the sector.
fn gr_decide_f64(qam: Qam, x: f64) -> f64 {
    let e = qam.energy().sqrt();
    #[allow(clippy::cast_possible_wrap, reason = "at most 32 levels")]
    let m = qam.levels_per_axis() as i64;
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let half = (m / 2) as f64;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the truncation of the C code"
    )]
    let s = ((x / (2.0 / e) + half) as i64).clamp(0, m - 1);
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let level = (2 * s - (m - 1)) as f64;
    level / e
}

/// Symbols of `qam` (random points) through the symbol-spaced `channel` (`(delay, magnitude,
/// phase)`), turned by `phase0 + 2 pi offset k`, plus uniform noise of `noise` per component.
fn qam_input(
    rng: &mut Pcg32,
    qam: Qam,
    layout: Layout,
    channel: &[(usize, f64, f64)],
    phase0: f64,
    offset: f64,
    noise: f64,
) -> Aligned<C32> {
    let n = layout.n;
    let skew = layout.skew(core::mem::size_of::<C32>());
    let symbols: Vec<C32> = (0..n).map(|_| qam.point(rng.next_u32())).collect();
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
// costas qam16 and qam64

/// The block defaults of `costas` (D604) with the constellation `qam16` or `qam64`.
const LOOP_BW: f64 = 0.02;
#[allow(clippy::approx_constant)]
const DAMPING: f64 = 0.7071;
const MAX_OFFSET: f64 = 0.05;
/// The start phase (rad) of 16-QAM and of 64-QAM (at 0.3 rad the outer 64-QAM points are
/// decided wrongly, D1425), the carrier offset (cycles per symbol) and the noise per component.
const CR_PHASE0_QAM16: f64 = 0.3;
const CR_PHASE0_QAM64: f64 = 0.05;
const CR_OFFSET: f64 = 0.0005;
const CR_NOISE: f64 = 0.05;
/// The start phase and the noise of 256-QAM and 1024-QAM (M44.2): the 64-QAM phase scaled by
/// the level spacing, and the 16-QAM noise in units of the level spacing (`0.05 sqrt(10 / E)`),
/// so that the loop locks and decides on the grid as for the smaller orders.
const CR_PHASE0_QAM256: f64 = 0.025;
const CR_PHASE0_QAM1024: f64 = 0.0125;
const CR_NOISE_QAM256: f64 = 0.012;
const CR_NOISE_QAM1024: f64 = 0.006;

fn costas_state(qam: Qam) -> CostasState {
    let c = match qam {
        Qam::Qam64 => CostasConstellation::Qam64,
        Qam::Qam256 => CostasConstellation::Qam256,
        Qam::Qam1024 => CostasConstellation::Qam1024,
        _ => CostasConstellation::Qam16,
    };
    CostasState::new(c, LOOP_BW, DAMPING, MAX_OFFSET).expect("costas")
}

/// GNU Radio's `constellation_receiver_cb` with the loop bandwidth of our loop constants and the
/// frequency limits of our `max_offset`.
fn gr_receiver(qam: Qam, points: &[[f32; 2]]) -> Option<GrConstellationReceiverCb> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the f32 parameters of the block"
    )]
    let (bw, f) = (
        (LOOP_BW / (DAMPING + 1.0 / (4.0 * DAMPING))) as f32,
        (core::f64::consts::TAU * MAX_OFFSET) as f32,
    );
    GrConstellationReceiverCb::new(gr_rect(qam, points), bw, -f, f)
}

/// The f64 model of the `constellation_receiver_cb` loop (exact sine and cosine, the sector
/// decision, `-arg(sample conj(d))`), the accuracy reference of [`Impl::GnuRadio`].
fn gr_receiver_model(qam: Qam, input: &[C32], alpha: f64, beta: f64) -> Vec<C32> {
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
        let (dr, di) = (gr_decide_f64(qam, yr), gr_decide_f64(qam, yi));
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

/// `costas` with a QAM constellation over `n` symbols.
pub(crate) struct CostasQamK {
    qam: Qam,
    input: Aligned<C32>,
    out: Aligned<C32>,
    points: Vec<[f32; 2]>,
    /// The entry: `costas_qam16_fallback`, `costas_qam64_fallback`, `costas_qam256_fallback` or
    /// `costas_qam1024_fallback`.
    fallback_entry: bool,
}

impl CostasQamK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam, fallback_entry: bool) -> Self {
        let skew = layout.skew(core::mem::size_of::<C32>());
        let (phase0, noise) = match qam {
            Qam::Qam64 => (CR_PHASE0_QAM64, CR_NOISE),
            Qam::Qam256 => (CR_PHASE0_QAM256, CR_NOISE_QAM256),
            Qam::Qam1024 => (CR_PHASE0_QAM1024, CR_NOISE_QAM1024),
            _ => (CR_PHASE0_QAM16, CR_NOISE),
        };
        CostasQamK {
            qam,
            input: qam_input(rng, qam, layout, &[(0, 1.0, 0.0)], phase0, CR_OFFSET, noise),
            out: Aligned::skewed(layout.n, skew),
            points: qam_points(qam),
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
        let mut state = costas_state(self.qam);
        let mut gr = if which == Impl::GnuRadio {
            gr_receiver(self.qam, &self.points)
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
                other => unreachable!("costas_qam has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let want = if which == Impl::GnuRadio {
            let (alpha, beta) =
                gr_receiver(self.qam, &self.points).map_or((0.0, 0.0), |g| g.gains());
            gr_receiver_model(self.qam, &self.input, f64::from(alpha), f64::from(beta))
        } else {
            let mut want = vec![C32::default(); self.input.len()];
            let _ = Reference.costas(&mut costas_state(self.qam), &self.input, &mut want);
            want
        };
        snr_db(&c32s(&want), &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours => !self.fallback_entry,
            Impl::Fallback => true,
            Impl::GnuRadio => self.fallback_entry && gr_receiver(self.qam, &self.points).is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => match self.qam {
                Qam::Qam64 => {
                    "GNU Radio constellation_receiver_cb with constellation_rect 64-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags)"
                }
                Qam::Qam256 => {
                    "GNU Radio constellation_receiver_cb with constellation_rect 256-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags)"
                }
                Qam::Qam1024 => {
                    "GNU Radio constellation_receiver_cb with constellation_rect 1024-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags)"
                }
                _ => {
                    "GNU Radio constellation_receiver_cb with constellation_rect 16-QAM (constellation_receiver_cb_impl.cc and constellation.cc of gr-digital, generic compiler flags)"
                }
            },
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// equalize qam16 and qam64

/// The taps of the block default.
const EQ_TAPS: usize = 11;
/// The carrier phase (rad) and the uniform noise per component of the workload.
const EQ_PHASE: f64 = 0.2;
const EQ_NOISE: f64 = 0.02;

/// The noise of the equalizer workload: [`EQ_NOISE`] for 16-QAM and 64-QAM, for 256-QAM and
/// 1024-QAM the 16-QAM noise in units of the level spacing (M44.2).
fn eq_noise(qam: Qam) -> f64 {
    match qam {
        Qam::Qam256 => 0.005,
        Qam::Qam1024 => 0.0025,
        _ => EQ_NOISE,
    }
}
/// The channel: `(delay in symbols, magnitude, phase)`.
const EQ_CHANNEL: [(usize, f64, f64); 3] = [(0, 1.0, 0.0), (1, 0.2, 0.6), (2, 0.05, -2.0)];

type EqState = EqualizeState<[C32; EQ_TAPS], [C32; EQ_TAPS]>;

fn eq_params(qam: Qam) -> EqParams {
    EqParams {
        constellation: match qam {
            Qam::Qam64 => EqConstellation::Qam64,
            Qam::Qam256 => EqConstellation::Qam256,
            Qam::Qam1024 => EqConstellation::Qam1024,
            _ => EqConstellation::Qam16,
        },
        ..EqParams::default()
    }
}

fn eq_state(qam: Qam) -> EqState {
    EqualizeState::new(
        [C32::default(); EQ_TAPS],
        [C32::default(); EQ_TAPS],
        eq_params(qam),
    )
    .expect("equalize")
}

/// The centre-spike taps of GNU Radio's equalizer (natural order).
fn gr_taps() -> [[f32; 2]; EQ_TAPS] {
    let mut t = [[0.0f32; 2]; EQ_TAPS];
    t[EQ_TAPS / 2] = [1.0, 0.0];
    t
}

/// GNU Radio's `linear_equalizer` on the `constellation_rect` of `qam` for `n` symbols per call
/// with the normalized decision-mode step of ours.
///
/// `volk` selects the build with VOLK's dispatcher (the reference of the SIMD entries
/// `equalize_qam16` and `equalize_qam64`: the dot products of the level, issue #169), otherwise
/// VOLK's generic dot products (the reference of the fallback entries).
fn gr_equalizer(
    qam: Qam,
    n: usize,
    points: &[[f32; 2]],
    volk: bool,
) -> Option<GrLinearEqualizerRect> {
    let (mu, rect) = (eq_params(qam).mu_dd, gr_rect(qam, points));
    if volk {
        GrLinearEqualizerRect::new_volk(&gr_taps(), mu, n, rect)
    } else {
        GrLinearEqualizerRect::new(&gr_taps(), mu, n, rect)
    }
}

/// The f64 model of GNU Radio's loop (the accuracy reference of [`Impl::GnuRadio`]): the FIR of
/// the window (oldest first) with the reversed taps, the sector decision, `e = d - y`,
/// `taps[i] += (mu / sum |x|^2) conj(x[i]) e`.
fn gr_equalizer_model(qam: Qam, input: &[C32]) -> Vec<C32> {
    let n = EQ_TAPS;
    let mu = f64::from(eq_params(qam).mu_dd);
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
        let (dr, di) = (gr_decide_f64(qam, yr), gr_decide_f64(qam, yi));
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

/// `equalize` with a QAM constellation over `n` symbols (see the module documentation).
pub(crate) struct EqualizeQamK {
    qam: Qam,
    input: Aligned<C32>,
    out: Aligned<C32>,
    points: Vec<[f32; 2]>,
    /// The state of the counted calls, carried across the runs.
    state: EqState,
    /// GNU Radio's equalizer of the counted calls (the fallback entry), carried across the runs.
    gr: Option<GrLinearEqualizerRect>,
    /// The entry: `equalize_qam16_fallback`, `equalize_qam64_fallback`,
    /// `equalize_qam256_fallback` or `equalize_qam1024_fallback`.
    fallback_entry: bool,
}

impl EqualizeQamK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam, fallback_entry: bool) -> Self {
        let skew = layout.skew(core::mem::size_of::<C32>());
        let points = qam_points(qam);
        let gr = gr_equalizer(qam, layout.n, &points, !fallback_entry);
        EqualizeQamK {
            qam,
            input: qam_input(rng, qam, layout, &EQ_CHANNEL, EQ_PHASE, 0.0, eq_noise(qam)),
            out: Aligned::skewed(layout.n, skew),
            points,
            state: eq_state(qam),
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
                other => unreachable!("equalize_qam has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The SNR of one call from the start state against the float32 model of `Reference` (the
    /// kernel is bit-exact, so every implementation gives the cap of [`snr_db`]); GNU Radio's
    /// loop against its f64 model.
    fn accuracy(&mut self, which: Impl) -> f64 {
        let mut got = vec![C32::default(); self.input.len()];
        let mut state = eq_state(self.qam);
        match which {
            Impl::Ours => equalize(&Sdrk, &mut state, &self.input, &mut got),
            Impl::Fallback => equalize(&Fallback, &mut state, &self.input, &mut got),
            Impl::GnuRadio => {
                if let Some(mut gr) = gr_equalizer(
                    self.qam,
                    self.input.len(),
                    &self.points,
                    !self.fallback_entry,
                ) {
                    gr.work(c32_pairs_ref(&self.input), c32_pairs(&mut got));
                }
                let want = gr_equalizer_model(self.qam, &self.input);
                return snr_db(&c32s(&want), &c32s(&got));
            }
            other => unreachable!("equalize_qam has no {other:?}"),
        }
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.equalize(&mut eq_state(self.qam), &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours => !self.fallback_entry,
            Impl::Fallback => true,
            Impl::GnuRadio => self.gr.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio if !self.fallback_entry => match self.qam {
                Qam::Qam64 => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter)"
                }
                Qam::Qam256 => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter)"
                }
                Qam::Qam1024 => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter)"
                }
                _ => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK's dispatched dot products of the level with the aligned tap copies of fir_filter)"
                }
            },
            Impl::GnuRadio => match self.qam {
                Qam::Qam64 => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 64-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags)"
                }
                Qam::Qam256 => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 256-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags)"
                }
                Qam::Qam1024 => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 1024-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags)"
                }
                _ => {
                    "GNU Radio linear_equalizer with adaptive_algorithm_nlms on constellation_rect 16-QAM (linear_equalizer_impl.cc and constellation.cc of gr-digital, VOLK generic dot products, generic compiler flags)"
                }
            },
            _ => "sdrk",
        }
    }
}

/// The samples per symbol of the block entries `qam_mod_block`, `qam_demod_block`,
/// `qam_mod_qam256_block` and `qam_demod_qam256_block`: the flows of M44 §5 and M44.2 §5 (25 kBd
/// at 200 kHz).
pub(crate) const BLOCK_SPS: usize = 8;

pub(crate) fn block_rate(hz: u64) -> Rate {
    Rate::hz(hz).expect("rate")
}

pub(crate) fn leaked_arena(bytes: usize) -> ArenaStorage<'static> {
    // The block lives as long as the measurement binary; its arena is leaked once.
    let mem: &'static mut [u8] = Box::leak(vec![0u8; bytes + 16].into_boxed_slice());
    ArenaStorage::new(mem)
}

/// The constellation of the blocks for `qam`.
fn block_constellation(qam: Qam) -> SymbolMapConstellation {
    match qam {
        Qam::Qpsk => SymbolMapConstellation::Qpsk,
        Qam::Qam16 => SymbolMapConstellation::Qam16,
        Qam::Qam64 => SymbolMapConstellation::Qam64,
        Qam::Qam256 => SymbolMapConstellation::Qam256,
        Qam::Qam1024 => SymbolMapConstellation::Qam1024,
        other => unreachable!("no block constellation for {other:?}"),
    }
}

fn qam_mod_config(qam: Qam) -> QamModConfig {
    QamModConfig {
        constellation: block_constellation(qam),
        samples_per_symbol: BLOCK_SPS,
        ..QamModConfig::default()
    }
}

fn qam_mod_block(qam: Qam) -> QamModBlock<ArenaStorage<'static>> {
    let cfg = qam_mod_config(qam);
    let rate = block_rate(100_000);
    let bytes = QamModBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    QamModBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("qam_mod block")
}

/// Random soft bits `+-1` of `n` symbols of `qam`.
fn qam_bits(rng: &mut Pcg32, qam: Qam, n: usize) -> Vec<f32> {
    (0..qam.bits_per_symbol() * n)
        .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// The block `qam_mod` (16-QAM for `qam_mod_block`, 256-QAM for `qam_mod_qam256_block`; 8
/// samples per symbol, roll-off 0.35, span 10) writing `n` output samples (`n / 8` symbols of
/// random bits) per call.
pub(crate) struct QamModBlockK {
    qam: Qam,
    block: QamModBlock<ArenaStorage<'static>>,
    bits: Vec<f32>,
    out: Vec<C32>,
    /// The libraries' modulator flow (the entries of issue #169, D2550).
    refs: ModRefs,
}

impl QamModBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam, liquid_entry: bool) -> Self {
        let bits = qam_bits(rng, qam, layout.n / BLOCK_SPS);
        let refs = ModRefs::new(&bits, qam.bits_per_symbol(), layout.n, liquid_entry);
        QamModBlockK {
            qam,
            block: qam_mod_block(qam),
            bits,
            out: vec![C32::default(); layout.n],
            refs,
        }
    }
}

#[inline(never)]
fn qam_mod_call(block: &mut QamModBlock<ArenaStorage<'static>>, bits: &[f32], out: &mut [C32]) {
    let _ = block.process(black_box(bits), out);
}

/// The GNU Radio `constellation_modulator` (generic_mod) prototype: the root-raised cosine with
/// roll-off `rolloff` at 32 samples per symbol (`nfilts` 32), `nfilts * 11 * sps` taps, which
/// `pfb_arb_resampler_ccf` splits into 32 branches of 88 taps. The gain makes its output the
/// output of `qam_mod` (the pulse of the block, `rrc(rolloff, sps, span) sqrt(sps)`, whose span
/// the prototype keeps: the taps beyond it are zero, so that the accuracy measures the
/// arithmetic of the filter and not another pulse); the centre
/// is where `qam_mod` has it, five symbols after the first tap, so that the outputs of the two
/// have the same time axis (the cost of the filter is that of GNU Radio's taps, all of them
/// are used).
const GR_MOD_NFILTS: usize = 32;

fn gr_mod_taps(rolloff: f64, sps: usize, span: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    let ntaps = GR_MOD_NFILTS * 11 * sps;
    let b = rolloff;
    let h = |t: f64| -> f64 {
        if t.abs() < 1e-9 {
            1.0 - b + 4.0 * b / PI
        } else if (t.abs() - 1.0 / (4.0 * b)).abs() < 1e-9 {
            b / 2f64.sqrt()
                * ((1.0 + 2.0 / PI) * (PI / (4.0 * b)).sin()
                    + (1.0 - 2.0 / PI) * (PI / (4.0 * b)).cos())
        } else {
            ((PI * t * (1.0 - b)).sin() + 4.0 * b * t * (PI * t * (1.0 + b)).cos())
                / (PI * t * (1.0 - (4.0 * b * t).powi(2)))
        }
    };
    // The peak of our pulse at the sample `span * sps / 2`.
    let mut ours = vec![0.0f64; sdrk_blocks::design::rrc_len(sps, span).expect("rrc")];
    sdrk_blocks::design::rrc(rolloff, sps, span, &mut ours).expect("rrc");
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let scale = ours[span * sps / 2] * (sps as f64).sqrt() / h(0.0);
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let centre = (span / 2) as f64;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "f32 taps of a library, small integers"
    )]
    (0..ntaps)
        .map(|x| {
            let t = x as f64 / GR_MOD_NFILTS as f64 - centre;
            // the pulse of the block ends at `span / 2` symbols on both sides: the taps beyond
            // are zero, but still multiplied by the 88 taps of every branch
            if t.abs() > centre { 0.0 } else { (h(t) * scale) as f32 }
        })
        .collect()
}

/// The comparison side of the modulator entries: GNU Radio's `constellation_modulator` as the
/// sum of its blocks (`pack_k_bits_bb` + `chunks_to_symbols_bc` + `pfb_arb_resampler_ccf`) or
/// liquid-dsp's `modemcf_modulate` + `firinterp_crcf`, on the bits of the block as bytes (made
/// once) and the points of its constellation.
struct ModRefs {
    k: usize,
    n_out: usize,
    liquid_entry: bool,
    bytes: Vec<u8>,
    symbols: Vec<[f32; 2]>,
    chunks: Option<GrChunksToSymbols>,
    pfb: Option<(GrPfbArb, bool)>,
    liquid: Option<(LiquidModemMap, LiquidFirinterpCrcf)>,
}

/// The points of the symbols `0..2^k` as the modulators make them from `k` bits (first the
/// bits of the in-phase axis, Gray coded; the quadrature axis after).
fn mod_points(k: usize) -> Vec<[f32; 2]> {
    (0..1usize << k)
        .map(|s| {
            let bits: Vec<f32> = (0..k)
                .map(|j| if (s >> (k - 1 - j)) & 1 == 1 { 1.0 } else { -1.0 })
                .collect();
            #[allow(clippy::cast_possible_truncation, reason = "f32 points of a table")]
            [
                axis_level_f64(&bits[..k / 2]) as f32,
                axis_level_f64(&bits[k / 2..]) as f32,
            ]
        })
        .collect()
}

impl ModRefs {
    fn new(bits: &[f32], k: usize, n_out: usize, liquid_entry: bool) -> Self {
        let sps = BLOCK_SPS;
        let n_sym = n_out / sps;
        let points = mod_points(k);
        let cfg = qam_mod_config(Qam::Qam16).pulse();
        let bytes: Vec<u8> = bits.iter().map(|&b| u8::from(b > 0.0)).collect();
        let mut r = ModRefs {
            k,
            n_out,
            liquid_entry,
            bytes,
            symbols: vec![[0.0; 2]; n_sym],
            chunks: None,
            pfb: None,
            liquid: None,
        };
        if liquid_entry {
            let taps = pulse_taps((sps as f64).sqrt());
            r.liquid = LiquidModemMap::new(&points, k)
                .zip(LiquidFirinterpCrcf::new(&taps, sps, true));
        } else {
            let taps = gr_mod_taps(cfg.rolloff, sps, cfg.span);
            r.chunks = GrChunksToSymbols::new(&points, k, n_sym);
            #[allow(clippy::cast_precision_loss, reason = "a small integer")]
            let rate = sps as f32;
            r.pfb = match GrPfbArb::new_volk(&taps, GR_MOD_NFILTS, rate, n_sym) {
                Some(b) => Some((b, true)),
                None => GrPfbArb::new(&taps, GR_MOD_NFILTS, rate, n_sym).map(|b| (b, false)),
            };
        }
        r
    }

    /// The same flow from the reset state (zero history) on the bits of `bits`.
    fn fresh(&self, bits: &[f32]) -> Self {
        ModRefs::new(bits, self.k, self.n_out, self.liquid_entry)
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::GnuRadio => self.chunks.is_some() && self.pfb.is_some(),
            Impl::Liquid => self.liquid.is_some(),
            _ => false,
        }
    }

    fn run(&mut self, which: Impl, out: &mut [C32]) {
        match which {
            Impl::GnuRadio => {
                let (Some(c), Some((p, _))) = (self.chunks.as_mut(), self.pfb.as_mut()) else {
                    unreachable!("gnuradio")
                };
                c.work(black_box(&self.bytes), &mut self.symbols);
                let _ = p.work(&self.symbols, c32_pairs(out));
            }
            Impl::Liquid => {
                let (m, f) = self.liquid.as_mut().expect("liquid");
                m.map(black_box(&self.bytes), &mut self.symbols);
                f.execute(&mut self.symbols, c32_pairs(out));
            }
            other => unreachable!("a modulator has no {other:?}"),
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::GnuRadio if self.pfb.as_ref().is_some_and(|p| p.1) => {
                "GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's dispatched dot product of the level"
            }
            Impl::GnuRadio => {
                "GNU Radio constellation_modulator as the sum of its blocks: pack_k_bits_bb + chunks_to_symbols_bc + pfb_arb_resampler_ccf (32 filters of 88 taps and their derivative filters), VOLK's generic dot product (the wasm32 build of VOLK has no SIMD)"
            }
            Impl::Liquid => {
                "liquid-dsp 1.8.3 modemcf_modulate (arbitrary-constellation modem) + firinterp_crcf (the root-raised cosine of the block), the SIMD dot products of the level (the portable loop at simd128)"
            }
            _ => "sdrk",
        }
    }
}

/// `qam_mod` in `f64` from a zero history: the points of M44.2 §1 (`(2 i - (m - 1)) / sqrt(E)`
/// per axis, `i` the level of the Gray code of the `b` bits of the axis; 16-QAM `(2 i - 3) /
/// sqrt(10)`) at every `sps`-th sample, filtered with `rrc(0.35, sps, 10) sqrt(sps)`.
fn qam_mod_f64(qam: Qam, bits: &[f32]) -> Vec<(f64, f64)> {
    let cfg = qam_mod_config(qam);
    let sps = cfg.samples_per_symbol;
    let mut taps = vec![0.0f64; sdrk_blocks::design::rrc_len(sps, cfg.span).expect("rrc")];
    sdrk_blocks::design::rrc(cfg.rolloff, sps, cfg.span, &mut taps).expect("rrc");
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let gain = (sps as f64).sqrt();
    let b = qam.bits_per_axis();
    let m = qam.levels_per_axis();
    let level = |bits: &[f32]| {
        let g = bits
            .iter()
            .fold(0usize, |g, &x| (g << 1) | usize::from(x > 0.0));
        let i = (0..m).find(|&i| i ^ (i >> 1) == g).unwrap_or(0);
        #[allow(clippy::cast_precision_loss, reason = "a small integer")]
        let v = (2 * i) as f64 - (m - 1) as f64;
        v / qam.energy().sqrt()
    };
    let syms: Vec<(f64, f64)> = bits
        .chunks_exact(2 * b)
        .map(|s| (level(&s[..b]), level(&s[b..])))
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
                Impl::Ours | Impl::Fallback => {
                    qam_mod_call(&mut self.block, &self.bits, &mut self.out)
                }
                other => self.refs.run(other, &mut self.out),
            }
            black_box(&self.out);
        }
    }

    /// One call from the reset state against the `f64` model.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        if matches!(which, Impl::GnuRadio | Impl::Liquid) {
            self.refs.fresh(&self.bits).run(which, &mut self.out);
        } else {
            self.block.reset();
            qam_mod_call(&mut self.block, &self.bits, &mut self.out);
        }
        let want: Vec<f64> = qam_mod_f64(self.qam, &self.bits)
            .into_iter()
            .flat_map(|(re, im)| [re, im])
            .collect();
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback) || self.refs.supports(which)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio | Impl::Liquid => self.refs.describe(which),
            _ => "sdrk",
        }
    }
}

fn psk_mod_config() -> PskModConfig {
    PskModConfig {
        constellation: SymbolMapConstellation::Qpsk,
        samples_per_symbol: BLOCK_SPS,
        ..PskModConfig::default()
    }
}

fn psk_mod_block() -> PskModBlock<ArenaStorage<'static>> {
    let cfg = psk_mod_config();
    let rate = block_rate(50_000);
    let bytes = PskModBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    PskModBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("psk_mod block")
}

/// The block `psk_mod` (coherent QPSK, 8 samples per symbol, roll-off 0.35, span 10) writing `n`
/// output samples (`n / 8` symbols of random bits) per call.
pub(crate) struct PskModBlockK {
    block: PskModBlock<ArenaStorage<'static>>,
    bits: Vec<f32>,
    out: Vec<C32>,
    /// The libraries' modulator flow (the entries of issue #169, D2550).
    refs: ModRefs,
}

impl PskModBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, liquid_entry: bool) -> Self {
        let bits: Vec<f32> = (0..2 * (layout.n / BLOCK_SPS))
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let refs = ModRefs::new(&bits, 2, layout.n, liquid_entry);
        PskModBlockK {
            block: psk_mod_block(),
            bits,
            out: vec![C32::default(); layout.n],
            refs,
        }
    }
}

#[inline(never)]
fn psk_mod_call(block: &mut PskModBlock<ArenaStorage<'static>>, bits: &[f32], out: &mut [C32]) {
    let _ = block.process(black_box(bits), out);
}

/// `psk_mod` with coherent QPSK in `f64` from a zero history: the Gray points of M40 at
/// `(1, 1) -> pi / 4`, `(0, 1) -> 3 pi / 4`, `(0, 0) -> 5 pi / 4`, `(1, 0) -> 7 pi / 4` at every
/// `sps`-th sample, filtered with `rrc(0.35, sps, 10) sqrt(sps)`.
fn psk_mod_f64(bits: &[f32]) -> Vec<(f64, f64)> {
    let cfg = psk_mod_config();
    let sps = cfg.samples_per_symbol;
    let mut taps = vec![0.0f64; sdrk_blocks::design::rrc_len(sps, cfg.span).expect("rrc")];
    sdrk_blocks::design::rrc(cfg.rolloff, sps, cfg.span, &mut taps).expect("rrc");
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let gain = (sps as f64).sqrt();
    let syms: Vec<(f64, f64)> = bits
        .chunks_exact(2)
        .map(|b| {
            let k = match (b[0] > 0.0, b[1] > 0.0) {
                (true, true) => 1.0,
                (false, true) => 3.0,
                (false, false) => 5.0,
                (true, false) => 7.0,
            };
            let a = k * std::f64::consts::FRAC_PI_4;
            (a.cos(), a.sin())
        })
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

impl Kernel for PskModBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    psk_mod_call(&mut self.block, &self.bits, &mut self.out);
                }
                other => self.refs.run(other, &mut self.out),
            }
            black_box(&self.out);
        }
    }

    /// One call from the reset state against the `f64` model.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        if matches!(which, Impl::GnuRadio | Impl::Liquid) {
            self.refs.fresh(&self.bits).run(which, &mut self.out);
        } else {
            self.block.reset();
            psk_mod_call(&mut self.block, &self.bits, &mut self.out);
        }
        let want: Vec<f64> = psk_mod_f64(&self.bits)
            .into_iter()
            .flat_map(|(re, im)| [re, im])
            .collect();
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        matches!(which, Impl::Ours | Impl::Fallback) || self.refs.supports(which)
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio | Impl::Liquid => self.refs.describe(which),
            _ => "sdrk",
        }
    }
}

fn qam_demod_block(qam: Qam) -> QamDemodBlock<ArenaStorage<'static>> {
    let cfg = QamDemodConfig {
        constellation: block_constellation(qam),
        symbol_rate_hz: 25_000.0,
        ..QamDemodConfig::default()
    };
    let rate = block_rate(200_000);
    let bytes = QamDemodBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    QamDemodBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("qam_demod block")
}

/// Calls of the warm-up of `qam_demod_block`: 6 calls of 512 symbols cover the 2000 blind symbols
/// of the equalizer (the multi-modulus start of 16-QAM, D2530; 1000 CMA symbols before). The
/// acquisition estimates at symbol 768, so the receiver is in `PullIn` until the first check at
/// symbol 4864 and in `Track` after it: with the warm-up call of the count the
/// counted calls take symbols 3584 to 11776, about 1280 of their 8192 symbols in `PullIn` (the
/// workload of M44, kept so that the cycles before and after M44.1 compare).
const DEMOD_WARMUP: usize = 6;

/// Calls of the warm-up of `qam_demod_qam256_block` (M44.2): 30 calls of 512 symbols (15360
/// symbols) cover the blind start of the 256-QAM equalizer (8000 multi-modulus symbols, D2509)
/// and the first check of the carrier loop (12288 symbols after the estimate at symbol 768, at
/// symbol 13056), so the receiver is in `Track` with the equalizer in decision mode; with the
/// warm-up call of the count the counted calls take symbols 15872 to 24064, before the next
/// check at 25344: the steady state of the receiver. The counted calls run no blind symbols:
/// they measure the steady state that follows the blind start.
const DEMOD_WARMUP_QAM256: usize = 30;

/// The windows of `n` input samples in the input of `qam_demod_qam256_block` (M44.2, D1524): the
/// signal is `8 n / 8` symbols long and every call reads the next window. The blind equalizer of
/// 256-QAM does not converge on 512 symbols repeated every call (the match of the decisions stays
/// at 5 to 46 % of the symbols over 34 calls, the MSE at 0.004, never locked; measured with the
/// workload of the entry), as it does not on the framed data of I44.2-6; with 4096 symbols it
/// locks within 16384 symbols and decides every symbol right.
const DEMOD_WINDOWS_QAM256: usize = 8;

/// The uniform noise per component of the input of the `qam_demod` entries: ±0.02 for 16-QAM,
/// the same in units of the level spacing for 256-QAM (`0.02 sqrt(10 / 170)`, about 0.005).
fn demod_noise(qam: Qam) -> f32 {
    match qam {
        Qam::Qam256 => 0.005,
        _ => 0.02,
    }
}

/// The input of `qam_demod_block` (and `pulse_match_block`, with the seed of its own entry; for
/// `qam_demod_qam256_block` with 256-QAM and `n` the length of all its windows): the soft bits of
/// `n / 8` random symbols of `qam` and the output of `qam_mod` for them, turned by 0.3 rad with
/// uniform noise of [`demod_noise`] per component.
fn qam_demod_input(rng: &mut Pcg32, qam: Qam, n: usize) -> (Vec<f32>, Vec<C32>) {
    let bits = qam_bits(rng, qam, n / BLOCK_SPS);
    let mut tx = qam_mod_block(qam);
    let mut sig = vec![C32::default(); n];
    // Two passes: the second starts with the pulse tails of the first, as the repeated input
    // does at every call.
    let _ = tx.process(&bits, &mut sig);
    let _ = tx.process(&bits, &mut sig);
    let (s, c) = 0.3f32.sin_cos();
    let noise = demod_noise(qam);
    let input = sig
        .iter()
        .map(|x| {
            C32::new(
                x.re * c - x.im * s + noise * rng.unit(),
                x.re * s + x.im * c + noise * rng.unit(),
            )
        })
        .collect();
    (bits, input)
}

/// The carrier loop bandwidth of `qam_demod` (its default `carrier_bw`) and the timing loop
/// bandwidth of 16-QAM (`QAM_SYNC_BW`), as GNU Radio's loops get them.
const DEMOD_CARRIER_BW: f64 = 0.0025;
const DEMOD_SYNC_BW: f32 = 0.005;

/// GNU Radio's receiver chain of `qam_demod_block` (16-QAM, issue #169, D2545): `fir_filter_ccf`
/// and `symbol_sync_cc` ([`GrFront`]), `linear_equalizer` with `adaptive_algorithm_nlms` on the
/// `constellation_rect` (decision-directed from the start: GNU Radio has no blind stage) and
/// `constellation_receiver_cb` with the carrier loop bandwidth of the block, as the sum of the
/// blocks on the same input; the output are the derotated symbols.
struct GrQamChain {
    front: GrFront,
    eq: GrLinearEqualizerRect,
    receiver: GrConstellationReceiverCb,
    equalized: Vec<[f32; 2]>,
    locked: Vec<[f32; 2]>,
}

impl GrQamChain {
    fn new(n: usize, points: &[[f32; 2]]) -> Option<Self> {
        let qam = Qam::Qam16;
        let cap = n / BLOCK_SPS + 64;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the f32 parameters of the block"
        )]
        let (bw, f) = (
            (DEMOD_CARRIER_BW / (DAMPING + 1.0 / (4.0 * DAMPING))) as f32,
            (core::f64::consts::TAU * MAX_OFFSET) as f32,
        );
        Some(GrQamChain {
            front: GrFront::new(n, DEMOD_SYNC_BW)?,
            eq: gr_equalizer(qam, cap, points, false)?,
            receiver: GrConstellationReceiverCb::new(gr_rect(qam, points), bw, -f, f)?,
            equalized: vec![[0.0; 2]; cap],
            locked: vec![[0.0; 2]; cap],
        })
    }

    /// One call: `input` through the four blocks; the derotated symbols are in `self.locked`,
    /// their number is returned.
    fn process(&mut self, input: &[[f32; 2]]) -> usize {
        let produced = self.front.process(input);
        self.eq.work(
            &self.front.symbols[..produced],
            &mut self.equalized[..produced],
        );
        self.receiver
            .work(&self.equalized[..produced], &mut self.locked[..produced]);
        produced
    }
}

#[inline(never)]
fn gr_qam_chain_call(chain: &mut GrQamChain, input: &[C32]) -> usize {
    chain.process(black_box(c32_pairs_ref(input)))
}

/// The block `qam_demod` (16-QAM for `qam_demod_block`, 256-QAM for `qam_demod_qam256_block`,
/// at 25 kBd and 200 kHz, the block defaults: 11 equalizer taps, `carrier_bw` 0.0025, `sync_bw`
/// 0.005 for 16-QAM and 0.002 for 256-QAM, D1527) reading `n` input samples per call: the
/// output of `qam_mod` for `n / 8` random symbols (repeated every call; `n / 8` is whole
/// symbols, so the symbol clock stays on the grid; for 256-QAM `DEMOD_WINDOWS_QAM256` windows of
/// `n` samples, one per call in turn), turned by 0.3 rad with uniform noise of [`demod_noise`]
/// per component. The state lives across the calls; the warm-up runs in `new`.
pub(crate) struct QamDemodBlockK {
    qam: Qam,
    block: QamDemodBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    /// The samples of a call.
    n: usize,
    /// The window of the next call.
    window: usize,
    symbols: Vec<u32>,
    out: Vec<f32>,
    produced: usize,
    /// GNU Radio's chain (16-QAM only) and the symbols of its last call.
    chain: Option<GrQamChain>,
    produced_gr: usize,
    points: Vec<[f32; 2]>,
}

#[inline(never)]
fn qam_demod_call(
    block: &mut QamDemodBlock<ArenaStorage<'static>>,
    input: &[C32],
    out: &mut [f32],
) -> usize {
    block
        .process(black_box(input), out)
        .map_or(0, |p| p.produced)
}

impl QamDemodBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, qam: Qam) -> Self {
        let n = layout.n;
        let bps = qam.bits_per_symbol();
        let windows = match qam {
            Qam::Qam256 => DEMOD_WINDOWS_QAM256,
            _ => 1,
        };
        let (bits, input) = qam_demod_input(rng, qam, n * windows);
        let symbols = bits
            .chunks_exact(bps)
            .map(|b| b.iter().fold(0u32, |g, &x| (g << 1) | u32::from(x > 0.0)))
            .collect();
        let warmup = match qam {
            Qam::Qam256 => DEMOD_WARMUP_QAM256,
            _ => DEMOD_WARMUP,
        };
        let mut k = QamDemodBlockK {
            qam,
            block: qam_demod_block(qam),
            input,
            n,
            window: 0,
            symbols,
            // Room for every soft bit of a call (a call may give a symbol more than nominal).
            out: vec![0.0; (n / BLOCK_SPS + 16) * bps],
            produced: 0,
            chain: None,
            produced_gr: 0,
            points: qam_points(qam),
        };
        if qam == Qam::Qam16 {
            k.chain = GrQamChain::new(n, &k.points);
        }
        for _ in 0..warmup {
            k.call();
            if let Some(chain) = k.chain.as_mut() {
                k.produced_gr = gr_qam_chain_call(chain, &k.input[..n]);
            }
        }
        k
    }

    /// One call on the next window.
    fn call(&mut self) {
        let at = self.window * self.n;
        self.window = (self.window + 1) % (self.input.len() / self.n);
        self.produced =
            qam_demod_call(&mut self.block, &self.input[at..at + self.n], &mut self.out);
    }
}

impl Kernel for QamDemodBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => self.call(),
                Impl::GnuRadio => {
                    let n = self.n;
                    let chain = self.chain.as_mut().expect("gnuradio");
                    self.produced_gr = gr_qam_chain_call(chain, &self.input[..n]);
                }
                other => unreachable!("qam_demod_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when the symbols of one call after the warm-up equal the sent ones at one of the four
    /// rotations and some lag (the input repeats, so the lag is taken modulo the symbols of the
    /// input), 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.run(which, 1);
        let got: Vec<u32> = if which == Impl::GnuRadio {
            // The derotated symbols of the chain, decided to the nearest point of the grid.
            let chain = self.chain.as_ref().expect("gnuradio");
            chain.locked[..self.produced_gr]
                .iter()
                .map(|s| nearest_point(&self.points, *s))
                .collect()
        } else {
            self.out[..self.produced]
                .chunks_exact(self.qam.bits_per_symbol())
                .map(|b| b.iter().fold(0u32, |g, &x| (g << 1) | u32::from(x > 0.0)))
                .collect()
        };
        #[allow(clippy::cast_possible_truncation, reason = "at most 5 bits per axis")]
        let b = self.qam.bits_per_axis() as u32;
        let m = self.symbols.len();
        let ok = got.len() + 2 >= self.n / BLOCK_SPS
            && (0..4u32).any(|r| {
                (0..m).any(|lag| {
                    got.iter().enumerate().all(|(j, &s)| {
                        let back = sdrk_core::kernels::constellation::rotate_symbol(s, b, 4 - r);
                        back == self.symbols[(j + m - lag) % m]
                    })
                })
            });
        if ok { 999.0 } else { 0.0 }
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => self.chain.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio fir_filter_ccf + symbol_sync_cc + linear_equalizer + constellation_receiver_cb (the sum of the blocks of the receiver chain, generic compiler flags and VOLK generic dot products; no acquisition, no blind equalizer stage and no lock watches)"
            }
            _ => "sdrk",
        }
    }
}

/// The index of the point of `points` nearest to `s` (the symbol of the grid).
fn nearest_point(points: &[[f32; 2]], s: [f32; 2]) -> u32 {
    let mut best = (f32::INFINITY, 0u32);
    for (i, p) in points.iter().enumerate() {
        let d = (p[0] - s[0]) * (p[0] - s[0]) + (p[1] - s[1]) * (p[1] - s[1]);
        if d < best.0 {
            #[allow(clippy::cast_possible_truncation, reason = "at most 1024 points")]
            {
                best = (d, i as u32);
            }
        }
    }
    best.1
}

/// The block `qam_demod` while its frequency acquisition searches (review of the QAM round,
/// D1422): the configuration of `qam_demod_block`, reading `n` input samples per call of
/// uniform noise of unit power (±1.2247 per component, no signal), repeated every call, after
/// the warm-up of `qam_demod_block`. No window of the fourth powers has a clear peak, so every
/// symbol goes through the acquisition's normalized fourth power and every 512 symbols through
/// its 1024-point `fft` and peak search, besides the matched filter, the timing loop, the
/// equalizer and the carrier loop of the steady state.
pub(crate) struct QamDemodSearchK {
    block: QamDemodBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<f32>,
}

impl QamDemodSearchK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let input = (0..n)
            .map(|_| C32::new(1.2247 * rng.unit(), 1.2247 * rng.unit()))
            .collect();
        let mut k = QamDemodSearchK {
            block: qam_demod_block(Qam::Qam16),
            input,
            out: vec![0.0; n / BLOCK_SPS * 4 + 64],
        };
        for _ in 0..DEMOD_WARMUP {
            qam_demod_call(&mut k.block, &k.input, &mut k.out);
        }
        k
    }
}

impl Kernel for QamDemodSearchK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    qam_demod_call(&mut self.block, &self.input, &mut self.out);
                }
                other => unreachable!("qam_demod_search_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when the acquisition is still searching after a call (no estimate taken from the
    /// noise), 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.run(which, 1);
        if self.block.acquired_frequency().is_none() {
            999.0
        } else {
            0.0
        }
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
                (0..4)
                    .rev()
                    .map(move |j| if (t >> j) & 1 == 1 { 1.0f32 } else { -1.0 })
            })
            .collect();
        let cfg = FrameDecodeConfig {
            rotations: FrameRotations::Four,
            constellation: SymbolMapConstellation::Qam16,
            ..FrameDecodeConfig::default()
        };
        let block =
            FrameDecodeBlock::new(&cfg, block_rate(100_000), &mut ArenaStorage::new(&mut []))
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
    block
        .process(black_box(soft), got)
        .map_or(0, |p| p.produced)
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

/// The constellation of a `symbol_map_block` entry: the families of the issue (PSK as `qpsk`,
/// 16-QAM, 64-QAM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MapKind {
    Qpsk,
    Qam16,
    Qam64,
}

impl MapKind {
    fn constellation(self) -> SymbolMapConstellation {
        match self {
            MapKind::Qpsk => SymbolMapConstellation::Qpsk,
            MapKind::Qam16 => SymbolMapConstellation::Qam16,
            MapKind::Qam64 => SymbolMapConstellation::Qam64,
        }
    }

    /// Bits per symbol.
    fn k(self) -> usize {
        match self {
            MapKind::Qpsk => 2,
            MapKind::Qam16 => 4,
            MapKind::Qam64 => 6,
        }
    }
}

/// The block `symbol_map` (`qpsk`, `qam16` or `qam64`; the first stage of `qam_mod` and
/// `psk_mod`, M44.1 §2.1) mapping the bits of `n` symbols per call to `n` points. The entries
/// `symbol_map_block*` are compared with GNU Radio's `pack_k_bits_bb` + `chunks_to_symbols_bc`
/// and the entries `symbol_map_block*_liquid` with liquid-dsp's `modemcf_modulate` on its
/// arbitrary-constellation modem (issue #169, D2548); both libraries get the bits as bytes (0
/// or 1), made once outside the counted calls, and the points of this block in its symbol order.
pub(crate) struct SymbolMapBlockK {
    kind: MapKind,
    block: SymbolMapBlock<ArenaStorage<'static>>,
    bits: Vec<f32>,
    out: Vec<C32>,
    /// The bits as bytes, one per byte (0 or 1), for the libraries.
    bytes: Vec<u8>,
    /// GNU Radio's mapper flow of the counted calls (the entries without `_liquid`).
    gr: Option<GrChunksToSymbols>,
    /// liquid's mapper of the counted calls (the entries with `_liquid`).
    liquid: Option<LiquidModemMap>,
}

/// The points of `kind` in the symbol order of the block: the block run on every symbol.
fn map_points(kind: MapKind) -> Vec<[f32; 2]> {
    let k = kind.k();
    let cfg = SymbolMapConfig {
        constellation: kind.constellation(),
        ..SymbolMapConfig::default()
    };
    let mut block = SymbolMapBlock::new(&cfg, block_rate(100_000), &mut ArenaStorage::new(&mut []))
        .expect("symbol_map block");
    let m = 1usize << k;
    let bits: Vec<f32> = (0..m)
        .flat_map(|s| {
            (0..k).map(move |j| {
                if (s >> (k - 1 - j)) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
        })
        .collect();
    let mut out = vec![C32::default(); m];
    let _ = block.process(&bits, &mut out);
    out.iter().map(|p| [p.re, p.im]).collect()
}

impl SymbolMapBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, kind: MapKind, liquid_entry: bool) -> Self {
        let cfg = SymbolMapConfig {
            constellation: kind.constellation(),
            ..SymbolMapConfig::default()
        };
        let block = SymbolMapBlock::new(&cfg, block_rate(100_000), &mut ArenaStorage::new(&mut []))
            .expect("symbol_map block");
        let k = kind.k();
        let bits: Vec<f32> = (0..k * layout.n)
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let bytes: Vec<u8> = bits.iter().map(|&b| u8::from(b > 0.0)).collect();
        let points = map_points(kind);
        SymbolMapBlockK {
            kind,
            block,
            bits,
            out: vec![C32::default(); layout.n],
            bytes,
            gr: if liquid_entry {
                None
            } else {
                GrChunksToSymbols::new(&points, k, layout.n)
            },
            liquid: if liquid_entry {
                LiquidModemMap::new(&points, k)
            } else {
                None
            },
        }
    }
}

#[inline(never)]
fn symbol_map_call(
    block: &mut SymbolMapBlock<ArenaStorage<'static>>,
    bits: &[f32],
    out: &mut [C32],
) {
    let _ = block.process(black_box(bits), out);
}

/// The level of one axis in `f64` from its bits (Gray, M44 §1): `(2 i - (m - 1)) / sqrt(E)` with
/// `m = 2^bits` levels, `E = 2 (m^2 - 1) / 3`, and `i` the level whose Gray code is the bits
/// (the most significant first); for `qpsk` (one bit per axis) `(2 b - 1) / sqrt(2)`.
fn axis_level_f64(b: &[f32]) -> f64 {
    let g = b
        .iter()
        .fold(0usize, |g, &x| (g << 1) | usize::from(x > 0.0));
    let m = 1usize << b.len();
    let i = (0..m).find(|&i| i ^ (i >> 1) == g).unwrap_or(0);
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let (v, e) = (
        (2 * i) as f64 - (m as f64 - 1.0),
        2.0 * ((m * m) as f64 - 1.0) / 3.0,
    );
    v / e.sqrt()
}

impl Kernel for SymbolMapBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    symbol_map_call(&mut self.block, &self.bits, &mut self.out);
                }
                Impl::GnuRadio => self
                    .gr
                    .as_mut()
                    .expect("gnuradio")
                    .work(black_box(&self.bytes), c32_pairs(&mut self.out)),
                Impl::Liquid => self
                    .liquid
                    .as_mut()
                    .expect("liquid")
                    .map(black_box(&self.bytes), c32_pairs(&mut self.out)),
                other => unreachable!("symbol_map_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// One call against the points of M44 §1 in `f64`.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.run(which, 1);
        let k = self.kind.k();
        let want: Vec<f64> = self
            .bits
            .chunks_exact(k)
            .flat_map(|b| [axis_level_f64(&b[..k / 2]), axis_level_f64(&b[k / 2..])])
            .collect();
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => self.gr.is_some(),
            Impl::Liquid => self.liquid.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio pack_k_bits_bb + chunks_to_symbols_bc (pack_k_bits.cc of gr-blocks, chunks_to_symbols_impl.cc of gr-digital), the table of the points of the block, generic compiler flags"
            }
            Impl::Liquid => {
                "liquid-dsp 1.8.3 modemcf_modulate on the arbitrary-constellation modem of the points of the block (modem_arb.proto.c), the bits packed by the wrapper"
            }
            _ => "sdrk",
        }
    }
}

/// The offset of `iq_offset_block` in samples: half a symbol at the 8 samples per symbol of the
/// flow `oqpsk_transmitter` (`psk_mod` with `offset`).
const IQ_OFFSET_SAMPLES: usize = BLOCK_SPS / 2;

/// The block `iq_offset` (`part` `imag`, `samples_per_symbol` 8: a delay of 4 samples; the last
/// stage of `psk_mod` with `offset`, M44.1 §2.4, M44.1.1 §3.3) over `n` samples of random values in ±1 per call, in place as in `psk_mod` (the
/// buffer is delayed again by every call; the copy of `Block::process` is a `memcpy` that glibc
/// does with `rep movsb` under the counters of `avx2` and `avx512`, which count every byte as one
/// execution, the artifact of the cycle tables).
pub(crate) struct IqOffsetBlockK {
    block: IqOffsetBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<C32>,
}

impl IqOffsetBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let cfg = IqOffsetConfig {
            samples_per_symbol: BLOCK_SPS,
            part: IqPart::Imag,
        };
        let rate = block_rate(100_000);
        let bytes =
            IqOffsetBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
        let input: Vec<C32> = (0..layout.n)
            .map(|_| C32::new(rng.unit(), rng.unit()))
            .collect();
        IqOffsetBlockK {
            block: IqOffsetBlock::new(&cfg, rate, &mut leaked_arena(bytes))
                .expect("iq_offset block"),
            out: input.clone(),
            input,
        }
    }
}

#[inline(never)]
fn iq_offset_call(block: &mut IqOffsetBlock<ArenaStorage<'static>>, data: &mut [C32]) {
    block.process_in_place(black_box(data));
}

impl Kernel for IqOffsetBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    iq_offset_call(&mut self.block, &mut self.out);
                }
                other => unreachable!("iq_offset_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when one call from the reset state gives the real parts unchanged and the imaginary
    /// parts 4 samples later (zero before), bit for bit; 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        self.out.copy_from_slice(&self.input);
        iq_offset_call(&mut self.block, &mut self.out);
        let ok = self.out.iter().enumerate().all(|(t, y)| {
            let im = t
                .checked_sub(IQ_OFFSET_SAMPLES)
                .map_or(0.0, |j| self.input[j].im);
            y.re.to_bits() == self.input[t].re.to_bits() && y.im.to_bits() == im.to_bits()
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

/// The block `qam_acquire` with `correct` `false` (the mode of `qam_demod`, M44.1 §2.2) while it
/// searches, in place as in `qam_demod` (`correct` `false` leaves the buffer as it is; the copy
/// of `Block::process` is the `rep movsb` artifact of `iq_offset_block`): `n` symbols of circular
/// complex Gaussian noise of unit power per call (no signal, the same every call), so every symbol goes through the normalized fourth power and
/// every 512 symbols through the 1024-point `fft` and the peak search; the share of the
/// acquisition in `qam_demod_search_block`.
pub(crate) struct QamAcquireBlockK {
    block: QamAcquireBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<C32>,
}

impl QamAcquireBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let cfg = QamAcquireConfig::default();
        let rate = block_rate(25_000);
        let bytes =
            QamAcquireBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
        // Circular complex Gaussian noise (Box-Muller): uniform noise in a square has a fourth
        // power with a mean, a clear peak at 0 that ends the search.
        let input: Vec<C32> = (0..layout.n)
            .map(|_| {
                let u = 0.5 * (rng.unit() + 1.0);
                let r = (-u.max(1e-12).ln()).sqrt();
                let (s, c) = (core::f32::consts::PI * rng.unit()).sin_cos();
                C32::new(r * c, r * s)
            })
            .collect();
        let mut k = QamAcquireBlockK {
            block: QamAcquireBlock::new(&cfg, rate, &mut leaked_arena(bytes))
                .expect("qam_acquire block"),
            out: input.clone(),
            input,
        };
        // Past the skip of the first search, as in the steady state of a search.
        qam_acquire_call(&mut k.block, &mut k.out);
        k
    }
}

#[inline(never)]
fn qam_acquire_call(block: &mut QamAcquireBlock<ArenaStorage<'static>>, data: &mut [C32]) {
    let _ = block.process_in_place(black_box(data));
}

impl Kernel for QamAcquireBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    qam_acquire_call(&mut self.block, &mut self.out);
                }
                other => unreachable!("qam_acquire_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when after a call the buffer is the input bit for bit (`correct` `false`) and the
    /// block still searches (no estimate taken from the noise), 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.run(which, 1);
        let same = self
            .out
            .iter()
            .zip(&self.input)
            .all(|(y, x)| y.re.to_bits() == x.re.to_bits() && y.im.to_bits() == x.im.to_bits());
        if same && self.block.estimate().is_none() && self.block.active() {
            999.0
        } else {
            0.0
        }
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

/// Payload and preamble bytes of the frames of `qam_frame_encode_block` and
/// `qam_frame_decode_block` (the frames of `frame_decode_block`).
const QAM_FRAME_PAYLOAD: usize = 32;
const QAM_FRAME_PREAMBLE: usize = 4;

/// A random packet of [`QAM_FRAME_PAYLOAD`] bytes.
fn qam_frame_packet(rng: &mut Pcg32) -> Packet {
    let payload: Vec<u8> = (0..QAM_FRAME_PAYLOAD)
        .map(|_| rng.next_u32() as u8)
        .collect();
    Packet::new(0, PacketFlags::NONE, &payload).expect("packet")
}

/// The bits of the frame of `p` as soft bits `+-1`.
fn qam_frame_soft(p: &Packet) -> Vec<f32> {
    let mut f = vec![0u8; frame::frame_bits(QAM_FRAME_PAYLOAD, QAM_FRAME_PREAMBLE)];
    frame::encode(p, QAM_FRAME_PREAMBLE, Whitening::None, &mut f).expect("frame");
    f.iter().map(|&b| if b == 1 { 1.0 } else { -1.0 }).collect()
}

/// The block `qam_frame_encode` (M44.1 §7: `frame_encode` -> `qam_mod`, 16-QAM at 8 samples per
/// symbol, 4-byte preamble, the flow `qam16_transmitter`) writing `n` output samples per call
/// from back-to-back frames of 32-byte payloads.
pub(crate) struct QamFrameEncodeBlockK {
    block: QamFrameEncodeBlock<ArenaStorage<'static>>,
    packets: Vec<Packet>,
    /// Packets consumed so far (the next call starts there; wraps around).
    next: usize,
    out: Vec<C32>,
}

impl QamFrameEncodeBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let cfg = QamFrameEncodeConfig {
            frame: FrameEncodeConfig {
                preamble_bytes: QAM_FRAME_PREAMBLE,
                ..FrameEncodeConfig::default()
            },
            qam: qam_mod_config(Qam::Qam16),
            // The workload of M44.1 (the flow `qam16_transmitter`); the scrambler of M44.2 has
            // its own entry, `bit_scramble_block`.
            scrambler: Scrambler::Off,
            ..QamFrameEncodeConfig::default()
        };
        let rate = block_rate(100_000);
        let bytes = QamFrameEncodeBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate)
            .expect("storage");
        let per = frame::frame_bits(QAM_FRAME_PAYLOAD, QAM_FRAME_PREAMBLE);
        // Enough packets for two calls, so the second call resumes a cut frame.
        let count = (2 * layout.n / BLOCK_SPS * 4).div_ceil(per) + 1;
        QamFrameEncodeBlockK {
            block: QamFrameEncodeBlock::new(&cfg, rate, &mut leaked_arena(bytes))
                .expect("qam_frame_encode block"),
            packets: (0..count).map(|_| qam_frame_packet(rng)).collect(),
            next: 0,
            out: vec![C32::default(); layout.n],
        }
    }
}

#[inline(never)]
fn qam_frame_encode_call(
    block: &mut QamFrameEncodeBlock<ArenaStorage<'static>>,
    packets: &[Packet],
    out: &mut [C32],
) -> usize {
    block
        .process(black_box(packets), out)
        .map_or(usize::MAX, |p| p.consumed)
}

impl Kernel for QamFrameEncodeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    if self.next >= self.packets.len() / 2 {
                        self.block.reset();
                        self.next = 0;
                    }
                    let consumed = qam_frame_encode_call(
                        &mut self.block,
                        &self.packets[self.next..],
                        &mut self.out,
                    );
                    self.next += consumed;
                }
                other => unreachable!("qam_frame_encode_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// One call from the reset state against `qam_mod` in `f64` (the model of `qam_mod_block`)
    /// of the bits of the frames back to back.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        self.next = 0;
        self.run(which, 1);
        let mut bits: Vec<f32> = self.packets.iter().flat_map(qam_frame_soft).collect();
        bits.truncate(self.out.len() / BLOCK_SPS * 4);
        let want: Vec<f64> = qam_mod_f64(Qam::Qam16, &bits)
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

/// The block `qam_frame_decode` (M44.1 §7: `qam_demod` with the configuration of
/// `qam_demod_block` -> `frame_decode` with `rotations` `four`, the frame watch at its default
/// 16384 symbols; the flow `qam16_receiver`) reading `n` input samples per call: the signal of
/// `qam_demod_block` with whole frames of 32-byte payloads back to back as the bits (the rest of
/// the `n / 8` symbols zero bits), repeated every call. The state lives across the calls; the
/// warm-up of `qam_demod_block` runs in `new`.
pub(crate) struct QamFrameDecodeBlockK {
    block: QamFrameDecodeBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    payloads: Vec<Vec<u8>>,
    got: Vec<Packet>,
    produced: usize,
}

#[inline(never)]
fn qam_frame_decode_call(
    block: &mut QamFrameDecodeBlock<ArenaStorage<'static>>,
    input: &[C32],
    got: &mut [Packet],
) -> usize {
    block
        .process(black_box(input), got)
        .map_or(0, |p| p.produced)
}

impl QamFrameDecodeBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let per = frame::frame_bits(QAM_FRAME_PAYLOAD, QAM_FRAME_PREAMBLE);
        let room = n / BLOCK_SPS * 4;
        let packets: Vec<Packet> = (0..room / per).map(|_| qam_frame_packet(rng)).collect();
        let mut bits: Vec<f32> = packets.iter().flat_map(qam_frame_soft).collect();
        bits.resize(room, -1.0);
        let mut tx = qam_mod_block(Qam::Qam16);
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
        let cfg = QamFrameDecodeConfig {
            demod: QamDemodConfig {
                constellation: SymbolMapConstellation::Qam16,
                symbol_rate_hz: 25_000.0,
                ..QamDemodConfig::default()
            },
            // The workload of M44.1 (the flow `qam16_receiver`); the descrambler of M44.2 has
            // its own entry, `bit_descramble_block`.
            scrambler: Scrambler::Off,
            ..QamFrameDecodeConfig::default()
        };
        let rate = block_rate(200_000);
        let bytes = QamFrameDecodeBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate)
            .expect("storage");
        let mut k = QamFrameDecodeBlockK {
            block: QamFrameDecodeBlock::new(&cfg, rate, &mut leaked_arena(bytes))
                .expect("qam_frame_decode block"),
            input,
            payloads: packets.iter().map(|p| p.data().to_vec()).collect(),
            // Room for every frame of a call (one frame is more than 56 soft bits).
            got: vec![Packet::default(); n / 56 + 1],
            produced: 0,
        };
        for _ in 0..DEMOD_WARMUP {
            k.produced = qam_frame_decode_call(&mut k.block, &k.input, &mut k.got);
        }
        k
    }
}

impl Kernel for QamFrameDecodeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    self.produced =
                        qam_frame_decode_call(&mut self.block, &self.input, &mut self.got);
                }
                other => unreachable!("qam_frame_decode_block has no {other:?}"),
            }
            black_box(&self.got);
        }
    }

    /// 999 when one call after the warm-up gives at least all frames of a call but one, every
    /// one with `CRC_OK` and a sent payload, 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.run(which, 1);
        let got = &self.got[..self.produced];
        let ok = got.len() + 1 >= self.payloads.len()
            && got.iter().all(|g| {
                g.flags.contains(PacketFlags::CRC_OK)
                    && self.payloads.iter().any(|p| g.data() == p.as_slice())
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

/// The `f32` taps of the pulse filters of the block entries (roll-off 0.35, span 10, 8 samples per
/// symbol: 81 taps): `rrc(0.35, 8, 10) * gain`, `gain` `sqrt(8)` for `pulse_shape`
/// and `1 / sqrt(8)` for `pulse_match`, each rounded once to `f32` as the blocks do.
fn pulse_taps(gain: f64) -> Vec<f32> {
    let cfg = qam_mod_config(Qam::Qam16).pulse();
    let mut taps = vec![0.0f64; sdrk_blocks::design::rrc_len(BLOCK_SPS, cfg.span).expect("rrc")];
    sdrk_blocks::design::rrc(cfg.rolloff, BLOCK_SPS, cfg.span, &mut taps).expect("rrc");
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the f32 taps of the blocks"
    )]
    taps.iter().map(|t| (t * gain) as f32).collect()
}

/// GNU Radio's `interp_fir_filter_ccf` / `fir_filter_ccf` (`interp` 1) with the taps of a block
/// entry for `n` input samples per call: the form with VOLK's dispatcher where the library has
/// it (the levels with SIMD), the scalar build otherwise (the wasm32 build of VOLK has only the
/// generic protokernels). The `bool` says which.
fn gr_fir_bank(taps: &[f32], interp: usize, n: usize) -> Option<(GrFirBank, bool)> {
    match GrFirBank::new_volk(taps, interp, n) {
        Some(b) => Some((b, true)),
        None => GrFirBank::new(taps, interp, n).map(|b| (b, false)),
    }
}

/// The block `pulse_shape` with the pulse of `qam_mod_block` (8 samples per symbol, roll-off
/// 0.35, span 10; the second stage of `qam_mod` and `psk_mod`, M44.1.1 §3.1) writing `n` output
/// samples per call from `n / 8` random 16-QAM points (the output of `symbol_map` on random
/// bits). The filter history lives across the calls.
pub(crate) struct PulseShapeBlockK {
    block: PulseShapeBlock<ArenaStorage<'static>>,
    bits: Vec<f32>,
    symbols: Vec<C32>,
    out: Vec<C32>,
    /// GNU Radio's `interp_fir_filter_ccf` of the counted calls (the entry `pulse_shape_block`),
    /// and whether it is the form with VOLK's dispatcher.
    gr: Option<(GrFirBank, bool)>,
    /// liquid's `firinterp_crcf` of the counted calls (the entry `pulse_shape_block_liquid`) with
    /// its input as pairs.
    liquid: Option<(LiquidFirinterpCrcf, Vec<[f32; 2]>)>,
}

impl PulseShapeBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, liquid_entry: bool) -> Self {
        let cfg = qam_mod_config(Qam::Qam16).pulse();
        let rate = block_rate(25_000);
        let bytes =
            PulseShapeBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
        let bits = qam_bits(rng, Qam::Qam16, layout.n / BLOCK_SPS);
        let map = SymbolMapConfig {
            constellation: SymbolMapConstellation::Qam16,
            ..SymbolMapConfig::default()
        };
        let mut mapper =
            SymbolMapBlock::new(&map, block_rate(100_000), &mut ArenaStorage::new(&mut []))
                .expect("symbol_map block");
        let mut symbols = vec![C32::default(); layout.n / BLOCK_SPS];
        let _ = mapper.process(&bits, &mut symbols);
        let taps = pulse_taps((BLOCK_SPS as f64).sqrt());
        let pairs = c32_pairs_ref(&symbols).to_vec();
        PulseShapeBlockK {
            block: PulseShapeBlock::new(&cfg, rate, &mut leaked_arena(bytes))
                .expect("pulse_shape block"),
            bits,
            symbols,
            out: vec![C32::default(); layout.n],
            gr: if liquid_entry {
                None
            } else {
                gr_fir_bank(&taps, BLOCK_SPS, layout.n / BLOCK_SPS)
            },
            liquid: if liquid_entry {
                LiquidFirinterpCrcf::new(&taps, BLOCK_SPS, true).map(|l| (l, pairs))
            } else {
                None
            },
        }
    }
}

#[inline(never)]
fn pulse_shape_call(
    block: &mut PulseShapeBlock<ArenaStorage<'static>>,
    symbols: &[C32],
    out: &mut [C32],
) {
    let _ = block.process(black_box(symbols), out);
}

impl Kernel for PulseShapeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    pulse_shape_call(&mut self.block, &self.symbols, &mut self.out);
                }
                Impl::GnuRadio => self
                    .gr
                    .as_mut()
                    .expect("gnuradio")
                    .0
                    .work(c32_pairs_ref(&self.symbols), c32_pairs(&mut self.out)),
                Impl::Liquid => {
                    let (l, x) = self.liquid.as_mut().expect("liquid");
                    l.execute(x, c32_pairs(&mut self.out));
                }
                other => unreachable!("pulse_shape_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// One call from the reset state against the `f64` model of `qam_mod` (the points and the
    /// pulse in `f64`).
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        if which == Impl::GnuRadio {
            // A fresh filter (zero history) on the same symbols.
            let taps = pulse_taps((BLOCK_SPS as f64).sqrt());
            if let Some((mut gr, _)) = gr_fir_bank(&taps, BLOCK_SPS, self.symbols.len()) {
                gr.work(c32_pairs_ref(&self.symbols), c32_pairs(&mut self.out));
            }
        } else if which == Impl::Liquid {
            let taps = pulse_taps((BLOCK_SPS as f64).sqrt());
            if let Some(mut l) = LiquidFirinterpCrcf::new(&taps, BLOCK_SPS, true) {
                let mut x = c32_pairs_ref(&self.symbols).to_vec();
                l.execute(&mut x, c32_pairs(&mut self.out));
            }
        } else {
            self.block.reset();
            pulse_shape_call(&mut self.block, &self.symbols, &mut self.out);
        }
        let want: Vec<f64> = qam_mod_f64(Qam::Qam16, &self.bits)
            .into_iter()
            .flat_map(|(re, im)| [re, im])
            .collect();
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => self.gr.is_some(),
            Impl::Liquid => self.liquid.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio if self.gr.as_ref().is_some_and(|g| g.1) => {
                "GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter)"
            }
            Impl::GnuRadio => {
                "GNU Radio interp_fir_filter_ccf (interp_fir_filter_impl.cc of gr-filter) with VOLK's generic dot product (the wasm32 build of VOLK has no SIMD), generic compiler flags"
            }
            Impl::Liquid => {
                "liquid-dsp 1.8.3 firinterp_crcf (firinterp.proto.c, firpfb.proto.c), the SIMD dot products of the level (the portable loop at simd128)"
            }
            _ => "sdrk",
        }
    }
}

/// The block `pulse_match` of `qam_demod_block` (25 kBd at 200 kHz, roll-off 0.35, span 10: 81
/// taps; the first stage of `qam_demod`, M44.1.1 §3.2) reading `n` input samples per call of an
/// input made like that of `qam_demod_block` (`qam_demod_input`, with the seed of this entry).
/// The filter history lives across the calls.
pub(crate) struct PulseMatchBlockK {
    block: PulseMatchBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<C32>,
    /// GNU Radio's `fir_filter_ccf` of the counted calls (the entry `pulse_match_block`), and
    /// whether it is the form with VOLK's dispatcher.
    gr: Option<(GrFirBank, bool)>,
    /// liquid's `firfilt_crcf` of the counted calls (the entry `pulse_match_block_liquid`) with
    /// its input as pairs.
    liquid: Option<(LiquidFirfiltCrcf, Vec<[f32; 2]>)>,
}

impl PulseMatchBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, liquid_entry: bool) -> Self {
        let n = layout.n;
        let (_, input) = qam_demod_input(rng, Qam::Qam16, n);
        let taps = pulse_taps(1.0 / (BLOCK_SPS as f64).sqrt());
        let pairs = c32_pairs_ref(&input).to_vec();
        let cfg = pulse_match_config();
        let rate = block_rate(200_000);
        let bytes =
            PulseMatchBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
        PulseMatchBlockK {
            block: PulseMatchBlock::new(&cfg, rate, &mut leaked_arena(bytes))
                .expect("pulse_match block"),
            out: vec![C32::default(); n],
            gr: if liquid_entry {
                None
            } else {
                gr_fir_bank(&taps, 1, n)
            },
            liquid: if liquid_entry {
                LiquidFirfiltCrcf::new(&taps, true).map(|l| (l, pairs))
            } else {
                None
            },
            input,
        }
    }
}

/// The matched filter of `qam_demod_block`: the pulse of its configuration.
fn pulse_match_config() -> PulseMatchConfig {
    QamDemodConfig {
        constellation: SymbolMapConstellation::Qam16,
        symbol_rate_hz: 25_000.0,
        ..QamDemodConfig::default()
    }
    .pulse()
}

#[inline(never)]
fn pulse_match_call(
    block: &mut PulseMatchBlock<ArenaStorage<'static>>,
    input: &[C32],
    out: &mut [C32],
) {
    let _ = block.process(black_box(input), out);
}

impl Kernel for PulseMatchBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    pulse_match_call(&mut self.block, &self.input, &mut self.out);
                }
                Impl::GnuRadio => self
                    .gr
                    .as_mut()
                    .expect("gnuradio")
                    .0
                    .work(c32_pairs_ref(&self.input), c32_pairs(&mut self.out)),
                Impl::Liquid => {
                    let (l, x) = self.liquid.as_mut().expect("liquid");
                    l.execute(x, c32_pairs(&mut self.out));
                }
                other => unreachable!("pulse_match_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// One call from the reset state against the filter in `f64` (the taps
    /// `rrc(0.35, 8, 10) / sqrt(8)` in `f64` on the same input, zeros before it).
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        if which == Impl::GnuRadio {
            // A fresh filter (zero history) on the same input.
            let taps = pulse_taps(1.0 / (BLOCK_SPS as f64).sqrt());
            if let Some((mut gr, _)) = gr_fir_bank(&taps, 1, self.input.len()) {
                gr.work(c32_pairs_ref(&self.input), c32_pairs(&mut self.out));
            }
        } else if which == Impl::Liquid {
            let taps = pulse_taps(1.0 / (BLOCK_SPS as f64).sqrt());
            if let Some(mut l) = LiquidFirfiltCrcf::new(&taps, true) {
                let mut x = c32_pairs_ref(&self.input).to_vec();
                l.execute(&mut x, c32_pairs(&mut self.out));
            }
        } else {
            self.block.reset();
            pulse_match_call(&mut self.block, &self.input, &mut self.out);
        }
        let cfg = pulse_match_config();
        let sps = BLOCK_SPS;
        let mut taps = vec![0.0f64; sdrk_blocks::design::rrc_len(sps, cfg.span).expect("rrc")];
        sdrk_blocks::design::rrc(cfg.rolloff, sps, cfg.span, &mut taps).expect("rrc");
        #[allow(clippy::cast_precision_loss, reason = "a small integer")]
        let gain = 1.0 / (sps as f64).sqrt();
        let want: Vec<f64> = (0..self.input.len())
            .flat_map(|t| {
                let (mut re, mut im) = (0.0, 0.0);
                for (k, h) in taps.iter().enumerate() {
                    if let Some(x) = t.checked_sub(k).map(|j| self.input[j]) {
                        re += h * gain * f64::from(x.re);
                        im += h * gain * f64::from(x.im);
                    }
                }
                [re, im]
            })
            .collect();
        snr_db(&want, &c32s(&self.out))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => self.gr.is_some(),
            Impl::Liquid => self.liquid.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio if self.gr.as_ref().is_some_and(|g| g.1) => {
                "GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's dispatched dot product of the level (volk_32fc_32f_dot_prod_32fc_a, the aligned tap copies of fir_filter)"
            }
            Impl::GnuRadio => {
                "GNU Radio fir_filter_ccf (fir_filter_ccf_impl.cc of gr-filter) with VOLK's generic dot product (the wasm32 build of VOLK has no SIMD), generic compiler flags"
            }
            Impl::Liquid => {
                "liquid-dsp 1.8.3 firfilt_crcf (firfilt.proto.c), the SIMD dot product of the level (the portable loop at simd128)"
            }
            _ => "sdrk",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Impl, Layout, prepare};

    /// Every entry of the building blocks and composites of M44.1 and M44.1.1 runs its workload as
    /// described: the accuracy check passes for ours and for the fallback.
    #[test]
    fn the_m44_1_block_entries_pass_their_accuracy_check() {
        let layout = Layout {
            n: 4096,
            misaligned: false,
        };
        for name in [
            "symbol_map_block",
            "iq_offset_block",
            "qam_acquire_block",
            "qam_frame_encode_block",
            "qam_frame_decode_block",
            "pulse_shape_block",
            "pulse_match_block",
        ] {
            for which in [Impl::Ours, Impl::Fallback] {
                let mut k = prepare(name, layout).expect(name);
                // Two calls more, as in the steady state of the count.
                k.run(which, 2);
                let a = k.accuracy(which);
                assert!(a >= 130.0, "{name} {}: {a}", which.name());
            }
        }
    }
}
