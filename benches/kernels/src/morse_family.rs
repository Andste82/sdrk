//! The Morse blocks (milestone M47): `morse_encode_block` and `morse_decode_block`.
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `morse_encode_block` | `sdrk_blocks::MorseEncodeBlock` | none (`info`) | one output sample (the key envelope at 8 kHz) |
//! | `morse_decode_block` | `sdrk_blocks::MorseDecodeBlock` | none (`info`) | one input sample (the envelope at 8 kHz) |
//!
//! Both are blocks without a kernel of their own: the encoder counts units and shapes the
//! edges, the decoder smooths, tracks two levels, merges short runs and measures runs in
//! scalar code, one sample after the other with the state of the sample before. No library has
//! these blocks, so both are counted against our own code as information (`Family::Own`), and
//! `ours` and `fallback` run the same code.
//!
//! Workloads: `morse_encode_block` fills 4096 samples per call from a long text packet at 20
//! words per minute and 8 kHz (5 ms edges), the same packet resubmitted until the call
//! consumes it, and starts the next after the word gap as a pipeline does; `morse_decode_block`
//! reads 4096 samples per call of the keyed envelope of six words `PARIS` after a lead-in of
//! silence, taken cyclically from a buffer of whole periods, so it keeps decoding words.

use std::hint::black_box;

use sdrk::Rate;
use sdrk_blocks::{
    ArenaStorage, Block, MorseDecodeBlock, MorseDecodeConfig, MorseEncodeBlock, MorseEncodeConfig,
    Packet, PacketFlags,
};

use super::{Impl, Kernel, Layout, Pcg32};

/// The sample rate of both workloads.
const FS: u64 = 8000;
/// The text of one period: six words.
const TEXT: &[u8] = b"PARIS PARIS PARIS PARIS PARIS PARIS";
/// Samples of silence in front of every period of the decoder workload (two units).
const LEAD: usize = 960;

fn packet() -> Packet {
    Packet::new(0, PacketFlags::NONE, TEXT).expect("packet")
}

fn encoder() -> MorseEncodeBlock<ArenaStorage<'static>> {
    MorseEncodeBlock::new(
        &MorseEncodeConfig::default(),
        Rate::hz(FS).expect("rate"),
        &mut ArenaStorage::new(&mut []),
    )
    .expect("morse_encode")
}

/// The block `morse_encode` over `n` output samples per call.
pub(crate) struct MorseEncodeBlockK {
    block: MorseEncodeBlock<ArenaStorage<'static>>,
    packet: Packet,
    out: Vec<f32>,
    n: usize,
}

impl MorseEncodeBlockK {
    pub(crate) fn new(_rng: &mut Pcg32, layout: Layout) -> Self {
        MorseEncodeBlockK {
            block: encoder(),
            packet: packet(),
            out: vec![0.0; layout.n],
            n: layout.n,
        }
    }
}

#[inline(never)]
fn encode_call(
    block: &mut MorseEncodeBlock<ArenaStorage<'static>>,
    packet: &[Packet],
    out: &mut [f32],
) -> usize {
    match block.process(black_box(packet), out) {
        Ok(p) => p.produced,
        Err(_) => 0,
    }
}

impl Kernel for MorseEncodeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    let produced = encode_call(
                        &mut self.block,
                        std::slice::from_ref(&self.packet),
                        &mut self.out[..self.n],
                    );
                    // A call that ends with the packet consumed starts the next call afresh.
                    if produced < self.n {
                        self.block.reset();
                    }
                }
                other => unreachable!("morse_encode_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when the first call from a reset gives the key envelope of the units of the text
    /// (hard edges are not used here, so the check is on the plateaus: the first unit of the
    /// word `P` is a dot of 480 samples that reaches 1.0).
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        self.run(which, 1);
        let ok = self.n >= 960
            && self.out[..5].iter().all(|&x| x < 1.0)
            && self.out[40..480].iter().all(|&x| x == 1.0)
            && self.out[520..960].iter().all(|&x| x < 0.01);
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

/// The block `morse_decode` over `n` envelope samples per call.
pub(crate) struct MorseDecodeBlockK {
    block: MorseDecodeBlock<ArenaStorage<'static>>,
    /// Whole periods plus `n` samples, so a call never wraps inside its input.
    env: Vec<f32>,
    period: usize,
    n: usize,
    at: usize,
    got: Vec<Packet>,
    produced: usize,
}

impl MorseDecodeBlockK {
    pub(crate) fn new(_rng: &mut Pcg32, layout: Layout) -> Self {
        // One period: the lead-in and the keyed words with their word gap.
        let mut enc = encoder();
        let mut keys = vec![0.0f32; 400_000];
        let p = enc
            .process(&[packet()], &mut keys)
            .expect("morse_encode")
            .produced;
        let mut env = vec![0.0f32; LEAD];
        env.extend_from_slice(&keys[..p]);
        let period = env.len();
        while env.len() < period + layout.n {
            env.extend_from_within(..period);
        }
        let block = MorseDecodeBlock::new(
            &MorseDecodeConfig::default(),
            Rate::hz(FS).expect("rate"),
            &mut ArenaStorage::new(&mut []),
        )
        .expect("morse_decode");
        MorseDecodeBlockK {
            block,
            env,
            period,
            n: layout.n,
            at: 0,
            // Room for every word of a call (a word needs at least 6 units at 40 words per minute).
            got: vec![Packet::default(); layout.n / 480 + 2],
            produced: 0,
        }
    }
}

#[inline(never)]
fn decode_call(
    block: &mut MorseDecodeBlock<ArenaStorage<'static>>,
    env: &[f32],
    got: &mut [Packet],
) -> usize {
    match block.process(black_box(env), got) {
        Ok(p) => p.produced,
        Err(_) => 0,
    }
}

impl Kernel for MorseDecodeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    let input = &self.env[self.at..self.at + self.n];
                    self.produced = decode_call(&mut self.block, input, &mut self.got);
                    self.at = (self.at + self.n) % self.period;
                }
                other => unreachable!("morse_decode_block has no {other:?}"),
            }
            black_box(&self.got);
        }
    }

    /// 999 when decoding the periods from a reset gives the six words `PARIS` of every period.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        self.at = 0;
        let mut words = 0;
        let mut right = true;
        let calls = 2 * self.period / self.n + 1;
        for _ in 0..calls {
            self.run(which, 1);
            for w in &self.got[..self.produced] {
                words += 1;
                // The first period starts with the learning block, which may cost the first word.
                right &= w.data() == b"PARIS";
            }
        }
        // Two periods hold twelve words; the first may be lost.
        if right && (11..=12).contains(&words) {
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
