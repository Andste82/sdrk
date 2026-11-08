//! The entries of milestone M45: the polyphase channelizer `channelize` (`C32`).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `channelize` | facade | liquid-dsp `firpfbch2_crcf` with the SIMD dot product of the level and FFTW (liquid's own FFT on wasm32) | our fallback |
//! | `channelize_fallback` | our fallback | liquid-dsp `firpfbch2_crcf` with the portable dot product and liquid's own FFT | — |
//! | `channelize_block` | the block `ChannelizeBlock` (the kernel through the facade) | none (`info`) | one input sample |
//! | `channel_select_block` | the block `ChannelSelectBlock` (channel -1 of 16 lanes) | none (`info`) | one output sample (one time step) |
//!
//! The two block entries (D928) are information: `channelize_block` is the kernel entry
//! `channelize` plus the block around it (the library comparison of the work is the kernel
//! entry, on the same workload), `channel_select_block` is a strided copy that no library has
//! as a block of its own; `ours` and `fallback` run the same block code.
//!
//! The workload is the default of the block `channelize`, the search grid of M45.1: 16 channels
//! with 8 taps per branch (a prototype of 128 taps from `design::channelizer_prototype` with the
//! edges 0.5 / 1.0 and 60 dB), 4096 input samples of uniform noise per call, so 512 hops of 16
//! channel samples (8192 output values). Both sides do the same work per hop: the newest 8
//! samples into the history (liquid: into its windows), 16 branch sums of 8 real taps over
//! complex samples, one 16-point transform; liquid scales the transform by `1 / M`, ours does
//! not (16 multiplications per hop more on liquid's side). liquid's analyzer is the 2x
//! oversampled M/2-hop bank of ours (`firpfbch2`: output rate `2 fs / M`, `L = 2 m` taps per
//! branch, the branch filters shifted by the output parity); its transform is the inverse one,
//! so its channel `k` is ours at `-k`. The state is carried across the runs (the steady state).
//!
//! Accuracy: ours against the f64 model of `Reference` from a fresh state, liquid against an f64
//! model of its own algorithm (the windows, the reversed branch filters, the inverse DFT and the
//! scale), so both are measured against the exact result of what they compute.

use std::hint::black_box;

use cref_sys::{Firpfbch2Build, LiquidFirpfbch2};
use sdrk::kernels::channelize::{Channelize, ChannelizeState, history_len};
use sdrk::{C32, Fallback, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, ChannelSelectBlock, ChannelSelectConfig, ChannelizeBlock, ChannelizeConfig, LanesMut,
    LanesRef,
};
use sdrk_core::Reference;

use super::{Aligned, Impl, Kernel, Layout, Pcg32, c32_pairs, c32s, noise_c32, snr_db};

/// The channels and the taps per branch of the workload (the block default).
const CHANNELS: usize = 16;
const TAPS_PER_BRANCH: usize = 8;

type State = ChannelizeState<C32, Vec<f32>, Vec<C32>, Vec<C32>>;

/// The prototype of the workload (time order): `channelizer_prototype` with the defaults of the
/// block, rounded to `f32`.
fn prototype() -> Vec<f32> {
    let mut h = vec![0.0f64; CHANNELS * TAPS_PER_BRANCH];
    sdrk_blocks::design::channelizer_prototype(CHANNELS, TAPS_PER_BRANCH, 0.5, 1.0, 60.0, &mut h)
        .expect("channelizer prototype");
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the f32 taps of the kernel"
    )]
    h.iter().map(|&v| v as f32).collect()
}

fn state(h: &[f32]) -> State {
    ChannelizeState::new(
        h.to_vec(),
        CHANNELS,
        vec![C32::default(); history_len(CHANNELS, TAPS_PER_BRANCH)],
        vec![C32::default(); CHANNELS / 2],
    )
    .expect("channelize state")
}

#[inline(never)]
fn channelize<B: Channelize<C32>>(backend: &B, s: &mut State, input: &[C32], out: &mut [C32]) {
    let _ = black_box(backend.channelize(s, black_box(input), out));
}

/// The f64 model of liquid's `firpfbch2_crcf` analyzer (see the module documentation): `M`
/// windows of `2 m` samples, the hop's `M / 2` samples pushed into the windows `base - 1 - i`
/// (`base = M / 2` on an even hop, `M` on an odd one), branch `i` filtered with the sub-filter
/// `(offset + i) mod M` (`offset = 0` / `M / 2`), whose tap `n` (newest sample first) is
/// `h[f + n M]`; then the inverse DFT divided by `M`.
fn liquid_model(h: &[f32], input: &[C32]) -> Vec<C32> {
    let m = CHANNELS;
    let d = m / 2;
    let sub = h.len() / m;
    let mut windows = vec![vec![(0.0f64, 0.0f64); sub]; m];
    let mut odd = false;
    let mut out = Vec::with_capacity(2 * input.len());
    for hop in input.chunks_exact(d) {
        let base = if odd { m } else { d };
        for (i, x) in hop.iter().enumerate() {
            let w = &mut windows[base - i - 1];
            w.rotate_left(1);
            w[sub - 1] = (f64::from(x.re), f64::from(x.im));
        }
        let offset = if odd { d } else { 0 };
        let sums: Vec<(f64, f64)> = (0..m)
            .map(|i| {
                let f = (offset + i) % m;
                let (mut re, mut im) = (0.0, 0.0);
                for n in 0..sub {
                    let t = f64::from(h[f + n * m]);
                    let (xr, xi) = windows[i][sub - 1 - n];
                    re += t * xr;
                    im += t * xi;
                }
                (re, im)
            })
            .collect();
        for k in 0..m {
            let (mut re, mut im) = (0.0, 0.0);
            for (n, &(xr, xi)) in sums.iter().enumerate() {
                let a = 2.0 * core::f64::consts::PI * ((k * n) % m) as f64 / m as f64;
                let (s, c) = a.sin_cos();
                re += xr * c - xi * s;
                im += xr * s + xi * c;
            }
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a model value rounded to f32"
            )]
            out.push(C32::new((re / m as f64) as f32, (im / m as f64) as f32));
        }
        odd = !odd;
    }
    out
}

/// `channelize` of `n` input samples (see the module documentation).
pub(crate) struct ChannelizeK {
    input: Aligned<C32>,
    out: Aligned<C32>,
    taps: Vec<f32>,
    /// The state of the counted calls, carried across the runs.
    state: State,
    /// liquid's analyzer of the counted calls, carried across the runs.
    liquid: Option<LiquidFirpfbch2>,
    /// The entry: `channelize_fallback`.
    fallback_entry: bool,
}

impl ChannelizeK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, fallback_entry: bool) -> Self {
        let skew = layout.skew(core::mem::size_of::<C32>());
        // Whole hops only, so that every call consumes all of its input.
        let n = layout.n / (CHANNELS / 2) * (CHANNELS / 2);
        let input = noise_c32(rng, n, 0.9, skew);
        let taps = prototype();
        let build = if fallback_entry {
            Firpfbch2Build::Portable
        } else if cfg!(target_arch = "wasm32") {
            // No FFTW build on wasm32: liquid's own FFT, with the dot product of the level
            // (the portable loop, liquid has no SIMD128 loop).
            Firpfbch2Build::Portable
        } else {
            Firpfbch2Build::Fftw
        };
        ChannelizeK {
            out: Aligned::skewed(2 * n, skew),
            state: state(&taps),
            liquid: LiquidFirpfbch2::new(&taps, CHANNELS, build),
            taps,
            input,
            fallback_entry,
        }
    }
}

impl Kernel for ChannelizeK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours => channelize(&Sdrk, &mut self.state, &self.input, &mut self.out),
                Impl::Fallback => {
                    channelize(&Fallback, &mut self.state, &self.input, &mut self.out);
                }
                Impl::Liquid => self
                    .liquid
                    .as_mut()
                    .expect("liquid")
                    .execute(c32_pairs(&mut self.input), c32_pairs(&mut self.out)),
                other => unreachable!("channelize has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The SNR of one call from a fresh state against the exact result of the same algorithm.
    fn accuracy(&mut self, which: Impl) -> f64 {
        let mut got = vec![C32::default(); 2 * self.input.len()];
        match which {
            Impl::Ours => channelize(&Sdrk, &mut state(&self.taps), &self.input, &mut got),
            Impl::Fallback => channelize(&Fallback, &mut state(&self.taps), &self.input, &mut got),
            Impl::Liquid => {
                let build = if self.fallback_entry || cfg!(target_arch = "wasm32") {
                    Firpfbch2Build::Portable
                } else {
                    Firpfbch2Build::Fftw
                };
                if let Some(mut q) = LiquidFirpfbch2::new(&self.taps, CHANNELS, build) {
                    let mut input = self.input.to_vec();
                    q.execute(c32_pairs(&mut input), c32_pairs(&mut got));
                }
                return snr_db(&c32s(&liquid_model(&self.taps, &self.input)), &c32s(&got));
            }
            other => unreachable!("channelize has no {other:?}"),
        }
        let mut want = vec![C32::default(); 2 * self.input.len()];
        let _ = Reference.channelize(&mut state(&self.taps), &self.input, &mut want);
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
            Impl::Liquid if self.fallback_entry || cfg!(target_arch = "wasm32") => {
                "liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), portable dot products, liquid's own FFT"
            }
            Impl::Liquid => {
                "liquid-dsp 1.8.3 firpfbch2_crcf analyzer (benches/refs/liquid-channelizer.c), the SIMD dot products of the level, FFTW 3.3.10"
            }
            _ => "sdrk",
        }
    }
}

/// The block `channelize` (the default configuration, the workload of `channelize`) over `n`
/// input samples per call: the kernel through the facade plus the block around it.
pub(crate) struct ChannelizeBlockK {
    block: ChannelizeBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    out: Vec<C32>,
}

fn channelize_block() -> ChannelizeBlock<ArenaStorage<'static>> {
    let cfg = ChannelizeConfig::default();
    let bytes = ChannelizeBlock::<ArenaStorage<'static>>::storage_bytes(&cfg).expect("storage") + 16;
    // The block lives as long as the measurement binary; its arena is leaked once.
    let mem: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
    ChannelizeBlock::new(&cfg, &mut ArenaStorage::new(mem)).expect("channelize block")
}

impl ChannelizeBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        // Whole hops only, so that every call consumes all of its input.
        let n = layout.n / (CHANNELS / 2) * (CHANNELS / 2);
        ChannelizeBlockK {
            block: channelize_block(),
            input: noise_c32(rng, n, 0.9, 0).to_vec(),
            out: vec![C32::default(); 2 * n],
        }
    }
}

#[inline(never)]
fn channelize_block_call(
    block: &mut ChannelizeBlock<ArenaStorage<'static>>,
    input: &[C32],
    out: &mut [C32],
) -> usize {
    match LanesMut::new(out, CHANNELS) {
        Ok(view) => block.process(black_box(input), view).map_or(0, |p| p.produced),
        Err(_) => 0,
    }
}

impl Kernel for ChannelizeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    channelize_block_call(&mut self.block, &self.input, &mut self.out);
                }
                other => unreachable!("channelize_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// The SNR of one call from a reset against the f64 model of `Reference` with the taps of
    /// the block.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        let mut got = vec![C32::default(); self.out.len()];
        let steps = channelize_block_call(&mut self.block, &self.input, &mut got);
        // The block holds the taps time-reversed; the state takes them in time order.
        let mut taps = self.block.taps().to_vec();
        taps.reverse();
        let mut want = vec![C32::default(); self.out.len()];
        let _ = Reference.channelize(&mut state(&taps), &self.input, &mut want);
        if steps * CHANNELS != want.len() {
            return 0.0;
        }
        snr_db(&c32s(&want), &c32s(&got))
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

/// The block `channel_select` (channel -1 of 16 lanes) over `n` time steps per call.
pub(crate) struct ChannelSelectBlockK {
    block: ChannelSelectBlock,
    lanes: Vec<C32>,
    out: Vec<C32>,
}

impl ChannelSelectBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let block = ChannelSelectBlock::new(&ChannelSelectConfig { channel: -1 }, CHANNELS)
            .expect("channel_select block");
        ChannelSelectBlockK {
            block,
            lanes: noise_c32(rng, layout.n * CHANNELS, 0.9, 0).to_vec(),
            out: vec![C32::default(); layout.n],
        }
    }
}

#[inline(never)]
fn channel_select_call(block: &mut ChannelSelectBlock, lanes: &[C32], out: &mut [C32]) -> usize {
    match LanesRef::new(black_box(lanes), CHANNELS) {
        Ok(view) => block.process(view, out).map_or(0, |p| p.produced),
        Err(_) => 0,
    }
}

impl Kernel for ChannelSelectBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    channel_select_call(&mut self.block, &self.lanes, &mut self.out);
                }
                other => unreachable!("channel_select_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when the output is lane 15 (channel -1) of every step, bit for bit.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let n = channel_select_call(&mut self.block, &self.lanes, &mut self.out);
        let lane = CHANNELS - 1;
        let ok = n == self.out.len()
            && self
                .out
                .iter()
                .zip(self.lanes.chunks_exact(CHANNELS))
                .all(|(o, step)| o.re.to_bits() == step[lane].re.to_bits()
                    && o.im.to_bits() == step[lane].im.to_bits());
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
