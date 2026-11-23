//! The entries of milestone M50: the overlap-save fast convolution `xcorr_fft` (`C32`).
//!
//! | Entry | Measured | Comparison | Information | Per |
//! | --- | --- | --- | --- | --- |
//! | `xcorr_fft_n512` | facade (`L` 127, `N` 512, `B` 386) | liquid-dsp `fftfilt_cccf`, `n` 512, FFTW (liquid's own FFT on wasm32) | our fallback | one output sample |
//! | `xcorr_fft_n512_fallback` | our fallback | liquid-dsp `fftfilt_cccf`, `n` 512, liquid's own FFT | — | one output sample |
//! | `xcorr_fft_n4096` | facade (`L` 1023, `N` 4096, `B` 3074) | liquid-dsp `fftfilt_cccf`, `n` 4096, FFTW (liquid's own FFT on wasm32) | our fallback | one output sample |
//! | `xcorr_fft_n4096_fallback` | our fallback | liquid-dsp `fftfilt_cccf`, `n` 4096, liquid's own FFT | — | one output sample |
//! | `correlate_block` | `sdrk_blocks::CorrelateBlock`, the default `prbs7` code (`L` 127, `N` 512) | none (`info`) | — | one output sample |
//!
//! A call takes the whole transform blocks that fit the layout's samples (10 blocks, 3860
//! samples, at `N` 512; one block, 3074 samples, at `N` 4096; the sample counts of the entries in
//! `xtask/src/count.rs`), so every call consumes its whole input; the state is carried across
//! the calls. The fallback entries are the pairs of the D321 cycle gate.
//!
//! The reference (M50 §5, D32): liquid's `fftfilt_cccf` (`benches/refs/liquid-fftfilt.c`) does
//! the same work, a streaming complex FFT fast convolution with complex taps in `f32`, by
//! overlap-add with a transform of `2 n` points per block of `n` outputs. Its block length is the
//! power of two nearest to our `B` (512 for 386, 4096 for 3074: liquid's transform of `2 n`
//! points is then twice ours, `N` 512 and 4096), with the same taps. liquid takes the same
//! samples per call as ours; the wrapper carries the partial block to the next call (D2619).
//!
//! The accuracy is the SNR of one call from a fresh state against the f64 direct convolution of
//! `Reference` (liquid: over the outputs of its whole blocks, from the call's input repeated
//! until liquid completes two blocks); `correlate_block` is 999 when one
//! call from a new block equals the kernel with the conjugated, reversed PRBS taps bit for bit.

use std::hint::black_box;

use cref_sys::{FftfiltBuild, LiquidFftfilt};
use sdrk::kernels::fft::FftPlan;
use sdrk::kernels::xcorr::{XcorrFft, XcorrFftState};
use sdrk::{C32, Fallback, Sdrk};
use sdrk_blocks::{ArenaStorage, Block, CorrelateBlock, CorrelateConfig};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32_pairs, c32s, noise_c32, snr_db};

type State = XcorrFftState<C32, Vec<C32>, Vec<C32>, Vec<C32>, Vec<C32>>;

fn state(taps: &[C32], n: usize) -> State {
    let plan = FftPlan::new(n, vec![C32::default(); n / 2]).expect("fft plan");
    XcorrFftState::new(
        taps,
        vec![C32::default(); n],
        vec![C32::default(); taps.len() - 1],
        vec![C32::default(); n],
        plan,
    )
    .expect("xcorr_fft state")
}

#[inline(never)]
fn xcorr<B: XcorrFft<C32>>(backend: &B, s: &mut State, input: &[C32], out: &mut [C32]) {
    let _ = black_box(backend.xcorr_fft(s, black_box(input), out));
}

/// `xcorr_fft` with `L` taps and an `N`-point FFT over whole blocks of the layout's samples.
pub(crate) struct XcorrK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    taps: Vec<C32>,
    n: usize,
    /// The state of the counted calls, carried across the runs.
    state: State,
    /// liquid's filter of the counted calls, carried across the runs.
    liquid: Option<LiquidFftfilt>,
    /// liquid's output (the samples of a call plus one block of carry).
    liquid_out: Vec<C32>,
    /// liquid's block length.
    liquid_block: usize,
    /// The entry: `<kernel>_fallback`.
    fallback_entry: bool,
}

/// liquid's build of the entry: liquid's own FFT for a fallback entry and on wasm32 (which has
/// no FFTW build), FFTW otherwise.
fn liquid_build(fallback_entry: bool) -> FftfiltBuild {
    if fallback_entry || cfg!(target_arch = "wasm32") {
        FftfiltBuild::Portable
    } else {
        FftfiltBuild::Fftw
    }
}

/// liquid's filter with the taps `taps` and the block length `block`.
fn liquid(taps: &[C32], block: usize, fallback_entry: bool) -> Option<LiquidFftfilt> {
    let h: Vec<[f32; 2]> = taps.iter().map(|v| [v.re, v.im]).collect();
    LiquidFftfilt::new(&h, block, liquid_build(fallback_entry))
}

impl XcorrK {
    pub(crate) fn new(
        rng: &mut Pcg32,
        layout: Layout,
        taps_len: usize,
        n: usize,
        liquid_block: usize,
        fallback_entry: bool,
    ) -> Self {
        let skew = layout.skew(core::mem::size_of::<C32>());
        let b = n - taps_len + 1;
        // Whole blocks only, so that every call consumes all of its input.
        let samples = (layout.n / b).max(1) * b;
        let input = noise_c32(rng, samples, 0.9, skew);
        #[allow(
            clippy::cast_precision_loss,
            reason = "the tap count as a scale factor"
        )]
        let g = 1.0 / (taps_len as f32).sqrt();
        let taps: Vec<C32> = noise_c32(rng, taps_len, 1.0, 0)
            .iter()
            .map(|v| C32::new(v.re * g, v.im * g))
            .collect();
        XcorrK {
            out: Aligned::skewed(samples, skew),
            state: state(&taps, n),
            liquid: liquid(&taps, liquid_block, fallback_entry),
            liquid_out: vec![C32::default(); samples + liquid_block],
            liquid_block,
            taps,
            n,
            input,
            fallback_entry,
        }
    }
}

impl Kernel for XcorrK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => xcorr(&Sdrk, &mut self.state, &self.input, &mut self.out),
                Impl::Fallback => xcorr(&Fallback, &mut self.state, &self.input, &mut self.out),
                Impl::Liquid => {
                    let _ = black_box(
                        self.liquid
                            .as_mut()
                            .expect("liquid")
                            .execute(c32_pairs(&mut self.input), c32_pairs(&mut self.liquid_out)),
                    );
                    black_box(&self.liquid_out);
                }
                other => unreachable!("xcorr_fft has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The SNR of one call from a fresh state against the f64 direct convolution.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let mut got = vec![C32::default(); self.input.len()];
        match which {
            Impl::Ours => xcorr(&Sdrk, &mut state(&self.taps, self.n), &self.input, &mut got),
            Impl::Liquid => {
                let Some(mut q) = liquid(&self.taps, self.liquid_block, self.fallback_entry) else {
                    return 0.0;
                };
                // The call's input repeated until liquid completes at least two of its blocks
                // (one call of `xcorr_fft_n4096` is shorter than one block of liquid's).
                let repeats = (2 * self.liquid_block).div_ceil(self.input.len());
                let mut input: Vec<C32> = (0..repeats)
                    .flat_map(|_| self.input.iter().copied())
                    .collect();
                let mut out = vec![C32::default(); input.len() + self.liquid_block];
                let produced = q.execute(c32_pairs(&mut input), c32_pairs(&mut out));
                let mut want = vec![C32::default(); input.len()];
                let _ = Reference.xcorr_fft(&mut state(&self.taps, self.n), &input, &mut want);
                return snr_db(&c32s(&want[..produced]), &c32s(&out[..produced]));
            }
            _ => xcorr(
                &Fallback,
                &mut state(&self.taps, self.n),
                &self.input,
                &mut got,
            ),
        }
        let mut want = vec![C32::default(); self.input.len()];
        let _ = Reference.xcorr_fft(&mut state(&self.taps, self.n), &self.input, &mut want);
        snr_db(&c32s(&want), &c32s(&got))
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours => !self.fallback_entry,
            Impl::Fallback => true,
            Impl::Liquid => self.liquid.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::Liquid if liquid_build(self.fallback_entry) == FftfiltBuild::Portable => {
                "liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), liquid's own FFT"
            }
            Impl::Liquid => {
                "liquid-dsp 1.8.3 fftfilt_cccf (benches/refs/liquid-fftfilt.c), FFTW 3.3.10"
            }
            _ => "sdrk",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The block

/// The block `correlate` (`CorrelateBlock`, the default configuration: `prbs7`, seed 1, one
/// sample per chip, `L` 127, `N` 512, `B` 386) over the whole blocks of the layout's samples, in
/// one call per run (the state carried across the runs).
pub(crate) struct CorrelateBlockK {
    block: CorrelateBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<C32>,
}

/// The block over storage that lives as long as the measurement (leaked once per entry).
fn correlate_block() -> CorrelateBlock<ArenaStorage<'static>> {
    let rate = sdrk::Rate::hz(48_000).expect("rate");
    let cfg = CorrelateConfig::default();
    let bytes = CorrelateBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate)
        .expect("correlate storage");
    // The alignment slack: the arena pads the first buffer to 16 bytes.
    let mem: &'static mut [u8] = Box::leak(vec![0u8; bytes + 16].into_boxed_slice());
    let mut arena = ArenaStorage::new(mem);
    CorrelateBlock::new(&cfg, rate, &mut arena).expect("correlate block")
}

impl CorrelateBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let block = correlate_block();
        let b = block.block_size();
        let samples = (layout.n / b).max(1) * b;
        let input = noise_c32(rng, samples, 0.9, 0).to_vec();
        CorrelateBlockK {
            block,
            out: vec![C32::default(); input.len()],
            input,
        }
    }
}

#[inline(never)]
fn correlate_call(
    block: &mut CorrelateBlock<ArenaStorage<'static>>,
    input: &[C32],
    out: &mut [C32],
) {
    let _ = black_box(block.process(black_box(input), out));
}

impl Kernel for CorrelateBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    correlate_call(&mut self.block, &self.input, &mut self.out);
                }
                other => unreachable!("correlate_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when one call from a new block equals the kernel with the taps
    /// `conj(r[L - 1 - k])` of the PRBS reference `r` (chip value `1 - 2b`), 0 otherwise.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let cfg = CorrelateConfig::default();
        let Ok(mut prbs) = sdrk_proto::prbs::Prbs::new(cfg.code, cfg.seed) else {
            return 0.0;
        };
        let reference: Vec<C32> = (0..127)
            .map(|_| C32::new(1.0 - 2.0 * f32::from(prbs.next_bit()), 0.0))
            .collect();
        let taps: Vec<C32> = reference.iter().rev().map(|v| v.conj()).collect();
        let mut want = vec![C32::default(); self.input.len()];
        xcorr(&Sdrk, &mut state(&taps, 512), &self.input, &mut want);
        let mut got = vec![C32::default(); self.input.len()];
        correlate_call(&mut correlate_block(), &self.input, &mut got);
        let same = want
            .iter()
            .zip(&got)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits());
        if same { 999.0 } else { 0.0 }
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
