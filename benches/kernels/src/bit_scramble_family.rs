//! The bit scramblers of M44.2 (the maintainer's decision on I44.2-6): `bit_scramble_block` and
//! `bit_descramble_block`.
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `bit_scramble_block` | `sdrk_blocks::BitScrambleBlock` | GNU Radio `scrambler_bb` | one bit |
//! | `bit_descramble_block` | `sdrk_blocks::BitDescrambleBlock` | GNU Radio `descrambler_bb` | one bit |
//!
//! Both blocks run the default configuration, the self-synchronizing scrambler
//! `1 + x^-18 + x^-23` (ITU-T V.32 to V.34) over plain bits (`constellation` `bpsk`), on 4096 soft
//! bits per call (random sign and magnitude), the register carried from call to call. The
//! blocks have no kernel and no backend, so `ours` and `fallback` run the same code. GNU Radio's
//! `scrambler_bb(0x21, 0, 22)` and `descrambler_bb(0x21, 0, 22)` (the same polynomial: the taps
//! at bits 0 and 5 of a register of 23 bits) run on the hard bits of the same input, unpacked
//! (one bit per byte), with GNU Radio's generic flags: the scalar reference of the work per bit.
//! GNU Radio's descrambler gives our bits; its scrambler gives them 23 bits later (its output is
//! the end of the register), the same work.

use std::hint::black_box;

use cref_sys::GrScrambler;
use sdrk::Rate;
use sdrk_blocks::{
    ArenaStorage, BitDescrambleBlock, BitScrambleBlock, BitScrambleConfig, Block,
};

use super::{Impl, Kernel, Layout, Pcg32};

/// The mask and the register length of GNU Radio for `1 + x^-18 + x^-23`.
const GR_MASK: u64 = 0x21;
const GR_LEN: u8 = 22;
/// The delays of the polynomial in bits.
const A: usize = 23;
const B: usize = 18;

/// The block of one direction.
enum Ours {
    Scramble(BitScrambleBlock<ArenaStorage<'static>>),
    Descramble(BitDescrambleBlock<ArenaStorage<'static>>),
}

impl Ours {
    fn new(descramble: bool) -> Self {
        let cfg = BitScrambleConfig::default();
        let rate = Rate::hz(9600).expect("rate");
        // The ring of the register: 23 `f32`.
        let mem: &'static mut [u8] = Box::leak(vec![0u8; 256].into_boxed_slice());
        let mut st = ArenaStorage::new(mem);
        if descramble {
            Ours::Descramble(BitDescrambleBlock::new(&cfg, rate, &mut st).expect("block"))
        } else {
            Ours::Scramble(BitScrambleBlock::new(&cfg, rate, &mut st).expect("block"))
        }
    }

    fn reset(&mut self) {
        match self {
            Ours::Scramble(b) => b.reset(),
            Ours::Descramble(b) => b.reset(),
        }
    }
}

#[inline(never)]
fn scramble_call(b: &mut BitScrambleBlock<ArenaStorage<'static>>, x: &[f32], y: &mut [f32]) {
    let _ = b.process(black_box(x), y);
}

#[inline(never)]
fn descramble_call(b: &mut BitDescrambleBlock<ArenaStorage<'static>>, x: &[f32], y: &mut [f32]) {
    let _ = b.process(black_box(x), y);
}

#[inline(never)]
fn gr_call(gr: &mut GrScrambler, x: &[u8], y: &mut [u8]) {
    let _ = gr.work(black_box(x), y);
}

/// `bit_scramble_block` (`descramble` false) or `bit_descramble_block` over `n` bits per call.
pub(crate) struct ScrambleBlockK {
    descramble: bool,
    ours: Ours,
    gr: Option<GrScrambler>,
    soft: Vec<f32>,
    bits: Vec<u8>,
    out: Vec<f32>,
    out_bits: Vec<u8>,
}

impl ScrambleBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, descramble: bool) -> Self {
        let soft: Vec<f32> = (0..layout.n)
            .map(|_| {
                let m = 0.05 + 1.95 * (rng.next_u32() as f32 / u32::MAX as f32);
                if rng.next_u32() & 1 == 1 { m } else { -m }
            })
            .collect();
        let bits = soft.iter().map(|&v| u8::from(v > 0.0)).collect();
        ScrambleBlockK {
            descramble,
            ours: Ours::new(descramble),
            gr: GrScrambler::new(GR_MASK, 0, GR_LEN, descramble),
            out: vec![0.0; layout.n],
            out_bits: vec![0; layout.n],
            soft,
            bits,
        }
    }

    /// The bits of the one-bit-at-a-time model of the register for one call from zero.
    fn model(&self) -> Vec<u8> {
        let mut reg: Vec<u8> = Vec::with_capacity(self.bits.len());
        let mut out = Vec::with_capacity(self.bits.len());
        for (n, &x) in self.bits.iter().enumerate() {
            let at = |reg: &Vec<u8>, d: usize| if n >= d { reg[n - d] } else { 0 };
            let y = x ^ at(&reg, A) ^ at(&reg, B);
            reg.push(if self.descramble { x } else { y });
            out.push(y);
        }
        out
    }
}

impl Kernel for ScrambleBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => match &mut self.ours {
                    Ours::Scramble(b) => scramble_call(b, &self.soft, &mut self.out),
                    Ours::Descramble(b) => descramble_call(b, &self.soft, &mut self.out),
                },
                Impl::GnuRadio => gr_call(
                    self.gr.as_mut().expect("gnuradio"),
                    &self.bits,
                    &mut self.out_bits,
                ),
                other => unreachable!("the bit scramblers have no {other:?}"),
            }
            black_box((&self.out, &self.out_bits));
        }
    }

    /// 999 when one call from the zero register gives the bits of the model (GNU Radio's
    /// scrambler 23 bits later); our magnitudes stay.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        let want = self.model();
        let ok = match which {
            Impl::GnuRadio => {
                let gr = self.gr.as_mut().expect("gnuradio");
                gr.reset();
                self.run(which, 1);
                let lag = if self.descramble { 0 } else { A };
                self.out_bits[lag..] == want[..want.len() - lag]
            }
            _ => {
                self.ours.reset();
                self.run(which, 1);
                self.out
                    .iter()
                    .zip(&self.soft)
                    .zip(&want)
                    .all(|((y, x), &w)| u8::from(*y > 0.0) == w && y.abs() == x.abs())
            }
        };
        if ok { 999.0 } else { 0.0 }
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => self.gr.is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match (which, self.descramble) {
            (Impl::Fallback, _) => "sdrk fallback",
            (Impl::GnuRadio, false) => {
                "GNU Radio scrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags)"
            }
            (Impl::GnuRadio, true) => {
                "GNU Radio descrambler_bb(0x21, 0, 22): gr::digital::lfsr, one __builtin_parityl per unpacked bit (generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}
