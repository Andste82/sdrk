//! The entries of milestone M45: the polyphase channelizer `channelize` (`C32`).
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `channelize` | facade | liquid-dsp `firpfbch2_crcf` with the SIMD dot product of the level and FFTW (liquid's own FFT on wasm32) | our fallback |
//! | `channelize_fallback` | our fallback | liquid-dsp `firpfbch2_crcf` with the portable dot product and liquid's own FFT | — |
//! | `channelize_block` | the block `ChannelizeBlock` (the kernel through the facade) | none (`info`) | one input sample |
//! | `channel_select_block` | the block `ChannelSelectBlock` (channel -1 of 16 lanes) | none (`info`) | one output sample (one time step) |
//! | `channel_bank_channelize` | eight PMR446 channels from 200 kS/s: `ChannelizeBlock` (16 channels, 24 taps per branch, 0.4 / 0.6) and eight `ChannelSelectBlock` (channels -8 to -1) | none (`info`) | one input sample |
//! | `channel_bank_ddc` | the same eight channels with eight `DdcBlock` (decimation 8, the -6 dB point at 6.25 kHz, transition 2.5 kHz, 60 dB) | none (`info`) | one input sample |
//!
//! The two bank entries are test 9 of M45: one channelizer with eight readers against eight
//! digital down-converters for the same channels (lanes -8 to -1 of 16, the channels 1 to 8 of
//! the flow `pmr446_monitor`), both on 4096 samples of uniform noise at 200 kS/s per call and
//! both giving eight channels at 25 kS/s with the same filter edges (flat to 0.4 of the spacing
//! of 12.5 kHz, 60 dB from 0.6; the Kaiser estimates give 291 and 290 taps). No threshold: the
//! cycles per input sample of both are the deliverable (`docs/performance.md`).
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
use sdrk::Rate;
use sdrk::kernels::channelize::{Channelize, ChannelizeState, history_len, taps_len};
use sdrk::kernels::decim::FirDecimState;
use sdrk::kernels::fused::Ddc;
use sdrk::kernels::mix::Nco;
use sdrk::{C32, Fallback, Sdrk};
use sdrk_blocks::{
    ArenaStorage, Block, ChannelSelectBlock, ChannelSelectConfig, ChannelizeBlock,
    ChannelizeConfig, DdcBlock, DdcConfig, LanesMut, LanesRef,
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

/// The tap storage of the state (`2 N` values) with the prototype `h` in its first half.
fn tap_storage(h: &[f32]) -> Vec<f32> {
    let mut t = vec![0.0f32; 2 * h.len()];
    t[..h.len()].copy_from_slice(h);
    t
}

/// The prototype in time order from the taps a state stores (time-reversed, every tap twice).
fn prototype_of(stored: &[f32]) -> Vec<f32> {
    let mut h: Vec<f32> = stored.iter().step_by(2).copied().collect();
    h.reverse();
    h
}

fn state(h: &[f32]) -> State {
    debug_assert_eq!(2 * h.len(), taps_len(CHANNELS, TAPS_PER_BRANCH));
    ChannelizeState::new(
        tap_storage(h),
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
    let bytes =
        ChannelizeBlock::<ArenaStorage<'static>>::storage_bytes(&cfg).expect("storage") + 16;
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
        Ok(view) => block
            .process(black_box(input), view)
            .map_or(0, |p| p.produced),
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
        // The block holds the taps time-reversed and twice each; the state takes the prototype
        // in time order.
        let taps = prototype_of(self.block.taps());
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
                .all(|(o, step)| {
                    o.re.to_bits() == step[lane].re.to_bits()
                        && o.im.to_bits() == step[lane].im.to_bits()
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

/// The channels of the bank entries: lanes -8 to -1 of 16 (PMR446 channels 1 to 8 with the
/// centre on channel 9).
const BANK_LANES: [i64; 8] = [-8, -7, -6, -5, -4, -3, -2, -1];
/// The input rate of the bank entries and the channel spacing.
const BANK_RATE_HZ: u32 = 200_000;
const BANK_SPACING_HZ: f64 = 12_500.0;
/// The decimation of the `ddc` bank: 200 kS/s to 25 kS/s, the step rate of the channelizer.
const BANK_DECIMATION: usize = 8;

/// The channelizer of the PMR446 plan (the node `chan` of `pmr446_monitor`).
fn pmr_config() -> ChannelizeConfig {
    ChannelizeConfig {
        channels: CHANNELS,
        taps_per_branch: 24,
        passband: 0.4,
        stopband: 0.6,
        ..ChannelizeConfig::default()
    }
}

/// An arena of `bytes` (plus the alignment slack) that lives as long as the measurement binary.
fn leaked_arena(bytes: usize) -> ArenaStorage<'static> {
    let mem: &'static mut [u8] = Box::leak(vec![0u8; bytes + 16].into_boxed_slice());
    ArenaStorage::new(mem)
}

/// `channel_bank_channelize`: one `channelize` block and eight `channel_select` blocks over `n`
/// input samples per call (see the module documentation).
pub(crate) struct ChannelBankK {
    chan: ChannelizeBlock<ArenaStorage<'static>>,
    selects: Vec<ChannelSelectBlock>,
    input: Vec<C32>,
    lanes: Vec<C32>,
    outs: Vec<Vec<C32>>,
}

impl ChannelBankK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n / (CHANNELS / 2) * (CHANNELS / 2);
        let cfg = pmr_config();
        let bytes = ChannelizeBlock::<ArenaStorage<'static>>::storage_bytes(&cfg).expect("storage");
        let chan = ChannelizeBlock::new(&cfg, &mut leaked_arena(bytes)).expect("channelize block");
        let selects = BANK_LANES
            .iter()
            .map(|&channel| {
                ChannelSelectBlock::new(&ChannelSelectConfig { channel }, CHANNELS)
                    .expect("channel_select block")
            })
            .collect();
        let steps = n / (CHANNELS / 2);
        ChannelBankK {
            chan,
            selects,
            input: noise_c32(rng, n, 0.9, 0).to_vec(),
            lanes: vec![C32::default(); steps * CHANNELS],
            outs: vec![vec![C32::default(); steps]; BANK_LANES.len()],
        }
    }
}

#[inline(never)]
fn channel_bank_call(k: &mut ChannelBankK) -> usize {
    let steps = channelize_block_call(&mut k.chan, black_box(&k.input), &mut k.lanes);
    let lanes = &k.lanes[..steps * CHANNELS];
    let mut produced = 0;
    for (block, out) in k.selects.iter_mut().zip(k.outs.iter_mut()) {
        produced += channel_select_call(block, lanes, out);
    }
    produced
}

impl Kernel for ChannelBankK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    channel_bank_call(self);
                }
                other => unreachable!("channel_bank_channelize has no {other:?}"),
            }
            black_box(&self.outs);
        }
    }

    /// The SNR of the eight channels of one call from a reset against the f64 model of
    /// `Reference` with the taps of the block.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.chan.reset();
        let produced = channel_bank_call(self);
        let taps = prototype_of(self.chan.taps());
        let steps = self.lanes.len() / CHANNELS;
        let mut state = ChannelizeState::new(
            tap_storage(&taps),
            CHANNELS,
            vec![C32::default(); history_len(CHANNELS, 24)],
            vec![C32::default(); CHANNELS / 2],
        )
        .expect("channelize state");
        let mut want = vec![C32::default(); self.lanes.len()];
        let _ = Reference.channelize(&mut state, &self.input, &mut want);
        if produced != steps * BANK_LANES.len() {
            return 0.0;
        }
        let (mut w, mut g) = (Vec::new(), Vec::new());
        for (&channel, out) in BANK_LANES.iter().zip(&self.outs) {
            let lane = channel.rem_euclid(CHANNELS as i64) as usize;
            w.extend(want.chunks_exact(CHANNELS).map(|step| step[lane]));
            g.extend_from_slice(out);
        }
        snr_db(&c32s(&w), &c32s(&g))
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

/// The `ddc` configuration of one channel of the bank: the channel filter of the channelizer
/// (flat to 5 kHz, 60 dB from 7.5 kHz) at the channel's offset.
fn bank_ddc_config(channel: i64) -> DdcConfig {
    DdcConfig {
        frequency_hz: channel as f64 * BANK_SPACING_HZ,
        decimation: BANK_DECIMATION,
        cutoff_hz: Some(6_250.0),
        transition_hz: Some(2_500.0),
        ..DdcConfig::default()
    }
}

fn bank_rate() -> Rate {
    Rate::hz(u64::from(BANK_RATE_HZ)).expect("rate")
}

/// `channel_bank_ddc`: eight `ddc` blocks over the same `n` input samples per call.
pub(crate) struct DdcBankK {
    ddcs: Vec<DdcBlock<C32, ArenaStorage<'static>>>,
    input: Vec<C32>,
    outs: Vec<Vec<C32>>,
}

impl DdcBankK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n / BANK_DECIMATION * BANK_DECIMATION;
        let rate = bank_rate();
        let ddcs = BANK_LANES
            .iter()
            .map(|&channel| {
                let cfg = bank_ddc_config(channel);
                let bytes = DdcBlock::<C32, ArenaStorage<'static>>::storage_bytes(&cfg, rate)
                    .expect("storage");
                DdcBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("ddc block")
            })
            .collect();
        DdcBankK {
            ddcs,
            input: noise_c32(rng, n, 0.9, 0).to_vec(),
            outs: vec![vec![C32::default(); n / BANK_DECIMATION]; BANK_LANES.len()],
        }
    }
}

#[inline(never)]
fn ddc_bank_call(k: &mut DdcBankK) -> usize {
    let mut produced = 0;
    for (block, out) in k.ddcs.iter_mut().zip(k.outs.iter_mut()) {
        produced += block
            .process(black_box(&k.input), out)
            .map_or(0, |p| p.produced);
    }
    produced
}

impl Kernel for DdcBankK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    ddc_bank_call(self);
                }
                other => unreachable!("channel_bank_ddc has no {other:?}"),
            }
            black_box(&self.outs);
        }
    }

    /// The SNR of the eight channels of one call from a reset against the kernel `ddc` of
    /// `Reference` (f64) with the taps of the design rounded to `f32` and the same oscillator.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        for block in &mut self.ddcs {
            block.reset();
        }
        let produced = ddc_bank_call(self);
        if produced != self.outs.len() * self.outs[0].len() {
            return 0.0;
        }
        let rate = bank_rate();
        let fs = f64::from(BANK_RATE_HZ);
        let (mut w, mut g) = (Vec::new(), Vec::new());
        for (&channel, out) in BANK_LANES.iter().zip(&self.outs) {
            let cfg = bank_ddc_config(channel);
            let spec = cfg.design_spec(rate);
            let n = spec.num_taps(fs, cfg.max_taps).expect("taps");
            let mut h = vec![0.0f64; n];
            spec.design(fs, cfg.max_taps, &mut h).expect("design");
            #[allow(clippy::cast_possible_truncation, reason = "the f32 taps of the block")]
            let taps: Vec<f32> = h.iter().map(|&v| v as f32).collect();
            let mut fir = FirDecimState::new(taps, vec![C32::default(); n - 1], BANK_DECIMATION)
                .expect("fir");
            let mut nco = Nco::new(-cfg.frequency_hz, rate).expect("nco");
            let mut want = vec![C32::default(); out.len()];
            let mut scratch = vec![C32::default(); self.input.len()];
            let _ = Reference.ddc(&mut nco, &mut fir, &self.input, &mut want, &mut scratch);
            w.extend_from_slice(&want);
            g.extend_from_slice(out);
        }
        snr_db(&c32s(&w), &c32s(&g))
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
