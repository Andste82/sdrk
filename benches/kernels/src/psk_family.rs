//! The PSK receiver block entry of issue #169 (M44.1 §8): `psk_demod_block`.
//!
//! | entry | measured | gate | information |
//! | --- | --- | --- | --- |
//! | `psk_demod_block` | `sdrk_blocks::PskDemodBlock`, QPSK | none (`info`) | GNU Radio's receiver chain |
//!
//! The block is `psk_demod` with coherent QPSK at 8 samples per symbol (25 kBd at 200 kHz), the
//! block defaults (roll-off 0.35, span 10, `sync_bw` 0.01, `carrier_bw` 0.02, `max_offset` 0.05,
//! no equalizer: the flow `qpsk_receiver`), reading 4096 input samples (512 symbols) per call
//! after a warm-up of 4 calls (the timing and carrier loops locked): the output of `psk_mod` for
//! random bits, turned by 0.3 rad with uniform noise of +-0.02 per component. A "sample" is an
//! input sample.
//!
//! GNU Radio has no such block; its receiver is a flowgraph. The comparison is the sum of the
//! blocks of that flowgraph on the same input (information, as the GNU Radio chain has no
//! acquisition and no lock watches): `fir_filter_ccf` with the taps of our matched filter (the
//! root-raised cosine of `psk_demod`), `symbol_sync_cc` (Gardner, MMSE 8-tap, `sync_bw` as the
//! loop bandwidth, damping 0.7071, `max_deviation` 0.005 symbols), `costas_loop_cc` (order 4,
//! the loop bandwidth of `carrier_bw` as for `costas_fallback`) and `constellation_decoder_cb`
//! with `constellation_qpsk` (`benches/refs/gr-symbol-sync.c`, `gr-costas.c`, `gr-psk-chain.c`).
//! The glue moves the samples between the blocks as GNU Radio's buffers do: the history of the
//! filter and the samples the timing loop leaves unconsumed are carried to the next call.
//!
//! The check of both sides is the bits (ours the signs of the soft bits, GNU Radio's the decoded
//! symbol) of one call after the warm-up against the sent bits at one of the four quarter turns
//! and some lag (the input repeats): 999 when they equal, 0 otherwise.

use std::hint::black_box;

use cref_sys::{GrCostasLoopCc, GrFirFilterCcf, GrSymbolSyncCc, gr_constellation_decoder_qpsk};
use sdrk::C32;
use sdrk_blocks::{
    ArenaStorage, Block, PskDemodBlock, PskDemodConfig, PskModBlock, PskModConfig,
    SymbolMapConstellation,
};

use super::qam_family::{BLOCK_SPS, block_rate, leaked_arena};
use super::{Impl, Kernel, Layout, Pcg32, c32_pairs_ref};

/// Calls of the warm-up (512 symbols each).
const WARMUP: usize = 4;
/// The symbol rate and sample rate of the block (8 samples per symbol).
const SYMBOL_RATE_HZ: f64 = 25_000.0;
const SAMPLE_RATE_HZ: u64 = 200_000;
/// The loop constants of the block defaults (`PskDemodConfig::default`).
const SYNC_BW: f32 = 0.01;
const CARRIER_BW: f64 = 0.02;
const DAMPING: f32 = 0.707;
const MAX_DEVIATION: f32 = 0.005;

fn demod_config() -> PskDemodConfig {
    PskDemodConfig {
        constellation: SymbolMapConstellation::Qpsk,
        symbol_rate_hz: SYMBOL_RATE_HZ,
        ..PskDemodConfig::default()
    }
}

fn psk_demod_block() -> PskDemodBlock<ArenaStorage<'static>> {
    let cfg = demod_config();
    let rate = block_rate(SAMPLE_RATE_HZ);
    let bytes = PskDemodBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    PskDemodBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("psk_demod block")
}

fn psk_mod_block() -> PskModBlock<ArenaStorage<'static>> {
    let cfg = PskModConfig {
        constellation: SymbolMapConstellation::Qpsk,
        samples_per_symbol: BLOCK_SPS,
        ..PskModConfig::default()
    };
    let rate = block_rate(50_000);
    let bytes = PskModBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate).expect("storage");
    PskModBlock::new(&cfg, rate, &mut leaked_arena(bytes)).expect("psk_mod block")
}

/// The matched filter of `psk_demod` (`rrc(0.35, 8, 10) / sqrt(8)`, `f32`).
pub(crate) fn matched_taps() -> Vec<f32> {
    let cfg = demod_config();
    let mut taps =
        vec![0.0f64; sdrk_blocks::design::rrc_len(BLOCK_SPS, cfg.span).expect("rrc length")];
    sdrk_blocks::design::rrc(cfg.rolloff, BLOCK_SPS, cfg.span, &mut taps).expect("rrc");
    #[allow(clippy::cast_precision_loss, reason = "a small integer")]
    let gain = 1.0 / (BLOCK_SPS as f64).sqrt();
    #[allow(clippy::cast_possible_truncation, reason = "the f32 taps of the block")]
    taps.iter().map(|t| (t * gain) as f32).collect()
}

/// The loop bandwidth of GNU Radio's `control_loop` that gives the gains of our carrier loop
/// (as `costas_fallback`).
fn gr_carrier_bw() -> f32 {
    let damping = f64::from(DAMPING);
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the f32 parameter of the block"
    )]
    {
        (CARRIER_BW / (damping + 1.0 / (4.0 * damping))) as f32
    }
}

/// The front end of GNU Radio's receiver chains: `fir_filter_ccf` with the matched filter, then
/// `symbol_sync_cc`, with the buffers the scheduler keeps between the calls (the history of the
/// filter and the samples the timing loop leaves unconsumed).
pub(crate) struct GrFront {
    fir: GrFirFilterCcf,
    sync: GrSymbolSyncCc,
    /// The last `ntaps - 1` input samples (the history of the filter) before the call's input.
    window: Vec<[f32; 2]>,
    /// The samples the timing loop has not consumed, then the filter output of the call.
    pending: Vec<[f32; 2]>,
    pending_len: usize,
    /// The symbols of the last call.
    pub(crate) symbols: Vec<[f32; 2]>,
}

impl GrFront {
    /// The front end for `n` input samples per call at 8 samples per symbol with the timing loop
    /// bandwidth `sync_bw` (damping 0.707, `max_deviation` 0.005 symbols, `ted_gain` 1).
    pub(crate) fn new(n: usize, sync_bw: f32) -> Option<Self> {
        let fir = GrFirFilterCcf::new(&matched_taps())?;
        #[allow(clippy::cast_precision_loss, reason = "a small integer")]
        let sps = BLOCK_SPS as f32;
        let sync = GrSymbolSyncCc::new(sps, sync_bw, DAMPING, 1.0, MAX_DEVIATION * sps)?;
        let history = fir.ntaps() - 1;
        Some(GrFront {
            fir,
            sync,
            window: vec![[0.0; 2]; history + n],
            pending: vec![[0.0; 2]; n + 64],
            pending_len: 0,
            symbols: vec![[0.0; 2]; n / BLOCK_SPS + 64],
        })
    }

    /// One call: `input` through the filter and the timing loop; the symbols are in
    /// [`GrFront::symbols`], their number is returned.
    pub(crate) fn process(&mut self, input: &[[f32; 2]]) -> usize {
        let history = self.fir.ntaps() - 1;
        let n = input.len();
        // The filter: the history, then the input.
        self.window.copy_within(n..n + history, 0);
        self.window[history..history + n].copy_from_slice(input);
        let at = self.pending_len;
        self.fir
            .work(&self.window[..history + n], &mut self.pending[at..at + n]);
        // The timing loop: the unconsumed samples of the call before, then the filter output.
        let (produced, consumed) = self.sync.work(&self.pending[..at + n], &mut self.symbols);
        self.pending.copy_within(consumed..at + n, 0);
        self.pending_len = at + n - consumed;
        produced
    }
}

/// GNU Radio's chain `fir_filter_ccf` + `symbol_sync_cc` + `costas_loop_cc` +
/// `constellation_decoder_cb` with its buffers.
struct GrChain {
    front: GrFront,
    costas: GrCostasLoopCc,
    locked: Vec<[f32; 2]>,
}

impl GrChain {
    fn new(n: usize) -> Option<Self> {
        Some(GrChain {
            front: GrFront::new(n, SYNC_BW)?,
            costas: GrCostasLoopCc::new(gr_carrier_bw(), 4)?,
            locked: vec![[0.0; 2]; n / BLOCK_SPS + 64],
        })
    }

    /// One call: `input` through the four blocks; the decoded symbols are written to `out`, the
    /// number of them is returned.
    fn process(&mut self, input: &[[f32; 2]], out: &mut [u8]) -> usize {
        let produced = self.front.process(input);
        // The carrier loop and the decision.
        self.costas.work(
            &self.front.symbols[..produced],
            &mut self.locked[..produced],
        );
        gr_constellation_decoder_qpsk(&self.locked[..produced], &mut out[..produced]);
        produced
    }
}

/// `psk_demod_block`: see the module documentation.
pub(crate) struct PskDemodBlockK {
    block: PskDemodBlock<ArenaStorage<'static>>,
    input: Vec<C32>,
    /// The sent bits (`+-1`, two per symbol: the sign of the real, then of the imaginary part).
    bits: Vec<f32>,
    out: Vec<f32>,
    produced: usize,
    chain: Option<GrChain>,
    decoded: Vec<u8>,
    produced_gr: usize,
}

#[inline(never)]
fn psk_demod_call(
    block: &mut PskDemodBlock<ArenaStorage<'static>>,
    input: &[C32],
    out: &mut [f32],
) -> usize {
    block
        .process(black_box(input), out)
        .map_or(0, |p| p.produced)
}

#[inline(never)]
fn gr_chain_call(chain: &mut GrChain, input: &[C32], out: &mut [u8]) -> usize {
    chain.process(black_box(c32_pairs_ref(input)), out)
}

impl PskDemodBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let n = layout.n;
        let bits: Vec<f32> = (0..2 * (n / BLOCK_SPS))
            .map(|_| if rng.next_u32() & 1 == 1 { 1.0 } else { -1.0 })
            .collect();
        let mut tx = psk_mod_block();
        let mut sig = vec![C32::default(); n];
        // Two passes: the second starts with the pulse tails of the first, as the repeated
        // input does at every call.
        let _ = tx.process(&bits, &mut sig);
        let _ = tx.process(&bits, &mut sig);
        let (s, c) = 0.3f32.sin_cos();
        let input: Vec<C32> = sig
            .iter()
            .map(|x| {
                C32::new(
                    x.re * c - x.im * s + 0.02 * rng.unit(),
                    x.re * s + x.im * c + 0.02 * rng.unit(),
                )
            })
            .collect();
        let mut k = PskDemodBlockK {
            block: psk_demod_block(),
            input,
            bits,
            out: vec![0.0; 2 * (n / BLOCK_SPS) + 64],
            produced: 0,
            chain: GrChain::new(n),
            decoded: vec![0; n / BLOCK_SPS + 64],
            produced_gr: 0,
        };
        for _ in 0..WARMUP {
            k.produced = psk_demod_call(&mut k.block, &k.input, &mut k.out);
            if let Some(chain) = k.chain.as_mut() {
                k.produced_gr = gr_chain_call(chain, &k.input, &mut k.decoded);
            }
        }
        k
    }

    /// 999 when `got` (pairs `(re > 0, im > 0)`) equals the sent bits at a quarter turn and a
    /// lag (modulo the call), 0 otherwise.
    fn check(&self, got: &[(bool, bool)]) -> f64 {
        let m = self.bits.len() / 2;
        let sent: Vec<(bool, bool)> = self
            .bits
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| (b[0] > 0.0, b[1] > 0.0))
            .collect();
        // One quarter turn of a point: (re, im) -> (-im, re).
        let turn = |(re, im): (bool, bool)| (!im, re);
        let ok = got.len() + 2 >= m
            && (0..4).any(|r| {
                (0..m).any(|lag| {
                    got.iter().enumerate().all(|(j, &g)| {
                        let mut g = g;
                        for _ in 0..r {
                            g = turn(g);
                        }
                        g == sent[(j + m - lag) % m]
                    })
                })
            });
        if ok { 999.0 } else { 0.0 }
    }
}

impl Kernel for PskDemodBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    self.produced = psk_demod_call(&mut self.block, &self.input, &mut self.out);
                }
                Impl::GnuRadio => {
                    let chain = self.chain.as_mut().expect("gnuradio");
                    self.produced_gr = gr_chain_call(chain, &self.input, &mut self.decoded);
                }
                other => unreachable!("psk_demod_block has no {other:?}"),
            }
            black_box(&self.out);
            black_box(&self.decoded);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        if which == Impl::GnuRadio {
            self.run(which, 1);
            let got: Vec<(bool, bool)> = self.decoded[..self.produced_gr]
                .iter()
                .map(|&s| (s & 1 == 1, s & 2 == 2))
                .collect();
            return self.check(&got);
        }
        self.run(which, 1);
        let got: Vec<(bool, bool)> = self.out[..self.produced]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| (b[0] > 0.0, b[1] > 0.0))
            .collect();
        self.check(&got)
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
                "GNU Radio fir_filter_ccf + symbol_sync_cc + costas_loop_cc + constellation_decoder_cb (the sum of the blocks of the receiver chain, generic compiler flags; no acquisition and no lock watches)"
            }
            _ => "sdrk",
        }
    }
}
