//! The RDS blocks (milestone M41): `rds_demod_block` and `rds_decode_block`.
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `rds_demod_block` | `sdrk_blocks::RdsDemodBlock` | none (`info`) | one input sample (the multiplex at 240 kHz) |
//! | `rds_decode_block` | `sdrk_blocks::RdsDecodeBlock` | none (`info`) | one input sample (one soft bit) |
//!
//! Both are composite blocks without a kernel of their own. `rds_demod` calls the kernels
//! `interleave`, `rotate`, `fir_decimate`, `symbol_sync` and `costas` through the facade (the
//! first three have SIMD paths, so the entry follows the level of the build) and decodes the
//! biphase chips in scalar code at 2375 chips per second; `rds_decode` is a bit-serial syndrome
//! recurrence. No library has these blocks, so both are counted against our own code as
//! information (`Family::Own`), and `ours` and `fallback` run the same code.
//!
//! Workloads: `rds_demod_block` reads 4096 samples per call of a 240 kHz multiplex (tones of
//! 1 and 2 kHz at 0.45, the pilot at 0.1 and the RDS subcarrier at 4 % with the biphase pulse
//! of IEC 62106 carrying groups 0A of PI `0xD123`), taken cyclically from a buffer of 0.5 s,
//! and keeps its state across calls as a pipeline does; `rds_decode_block` reads 4096 `+-1.0`
//! soft bits per call of the same groups, continued across calls (the period of 40 groups is
//! a multiple of the group), so it stays synchronized and outputs a packet every 104 samples.

use std::f64::consts::PI;
use std::hint::black_box;

use sdrk::Rate;
use sdrk_blocks::{
    ArenaStorage, Block, Packet, PacketFlags, RdsDecodeBlock, RdsDecodeConfig, RdsDemodBlock,
    RdsDemodConfig,
};
use sdrk_proto::rds::{self, Offset};

use super::{Impl, Kernel, Layout, Pcg32};

/// The multiplex rate of the demodulator workload.
const FS: u64 = 240_000;
/// Multiplex samples of the cyclic buffer (0.5 s).
const MPX_LEN: usize = 120_000;
/// Groups of the period of the decoder workload.
const PERIOD_GROUPS: usize = 40;

/// The data bits of `groups` random 0A groups of PI `0xD123` (random block D).
fn group_bits(rng: &mut Pcg32, groups: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(groups * 104);
    for g in 0..groups {
        let words = [
            0xD123,
            0x0148 | (g as u16 & 3),
            0xE0E0,
            rng.next_u32() as u16,
        ];
        for (w, o) in words
            .iter()
            .zip([Offset::A, Offset::B, Offset::C, Offset::D])
        {
            let block = rds::block26(*w, o);
            bits.extend((0..26).rev().map(|i| ((block >> i) & 1) as u8));
        }
    }
    bits
}

/// `h(x)` of the biphase pulse (`x = t / Tb`, the closed form of `rds.py`).
fn h(x: f64) -> f64 {
    let d = 1.0 - 64.0 * x * x;
    if d.abs() < 1e-12 {
        return 2.0;
    }
    8.0 * (4.0 * PI * x).cos() / (PI * d)
}

/// `MPX_LEN` samples of the multiplex with the differentially coded `bits` on the subcarrier.
fn multiplex(bits: &[u8]) -> Vec<f32> {
    let tb = 1.0 / 1187.5;
    let mut t_bit = 0u8;
    let levels: Vec<f64> = bits
        .iter()
        .map(|&d| {
            t_bit ^= d;
            2.0 * f64::from(t_bit) - 1.0
        })
        .collect();
    let pulse = |x: f64| {
        if (-2.0..2.0).contains(&x) {
            (h(x + 0.25) - h(x - 0.25)) / 2.72
        } else {
            0.0
        }
    };
    (0..MPX_LEN)
        .map(|m| {
            let t = m as f64 / FS as f64;
            let k0 = (t / tb).floor() as i64;
            let mut b = 0.0;
            for k in k0 - 2..=k0 + 2 {
                if let Some(a) = usize::try_from(k).ok().and_then(|k| levels.get(k)) {
                    b += a * pulse(t / tb - k as f64);
                }
            }
            let phi = 2.0 * PI * 19_000.0 * t;
            let l = (2.0 * PI * 1000.0 * t).sin();
            let r = (2.0 * PI * 2000.0 * t).sin();
            (0.45 * (l + r)
                + 0.45 * (l - r) * (2.0 * phi).sin()
                + 0.1 * phi.sin()
                + 0.04 * b * (3.0 * phi).sin()) as f32
        })
        .collect()
}

/// The block `rds_demod` over `n` multiplex samples per call.
pub(crate) struct RdsDemodBlockK {
    block: RdsDemodBlock<ArenaStorage<'static>>,
    mpx: Vec<f32>,
    bits: Vec<u8>,
    n: usize,
    at: usize,
    out: Vec<f32>,
}

impl RdsDemodBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let groups = (MPX_LEN as f64 / FS as f64 * 1187.5 / 104.0).ceil() as usize + 1;
        let bits = group_bits(rng, groups);
        let mpx = multiplex(&bits);
        let cfg = RdsDemodConfig::default();
        let rate = Rate::hz(FS).expect("rate");
        let bytes = RdsDemodBlock::<ArenaStorage<'static>>::storage_bytes(&cfg, rate)
            .expect("storage")
            + 16;
        // The block lives as long as the measurement binary; its arena is leaked once.
        let mem: &'static mut [u8] = Box::leak(vec![0u8; bytes].into_boxed_slice());
        let block = RdsDemodBlock::new(&cfg, rate, &mut ArenaStorage::new(mem)).expect("rds_demod");
        RdsDemodBlockK {
            block,
            mpx,
            bits,
            n: layout.n,
            at: 0,
            out: vec![0.0; layout.n / 100 + 64],
        }
    }
}

#[inline(never)]
fn demod_samples(
    block: &mut RdsDemodBlock<ArenaStorage<'static>>,
    input: &[f32],
    out: &mut [f32],
) -> usize {
    match block.process(black_box(input), out) {
        Ok(p) => p.consumed,
        Err(_) => 0,
    }
}

impl Kernel for RdsDemodBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    if self.at + self.n > self.mpx.len() {
                        self.at = 0;
                    }
                    let mut rest = &self.mpx[self.at..self.at + self.n];
                    while !rest.is_empty() {
                        let c = demod_samples(&mut self.block, rest, &mut self.out);
                        if c == 0 {
                            break;
                        }
                        rest = &rest[c..];
                    }
                    self.at += self.n;
                }
                other => unreachable!("rds_demod_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    /// 999 when the whole buffer, from a reset, gives the data bits without an error after the
    /// first 100 bits (at the bit lag 0, D669), otherwise 0.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        let mut soft = Vec::new();
        let mut room = vec![0.0f32; 1024];
        let mut rest = &self.mpx[..];
        while !rest.is_empty() {
            let p = self.block.process(rest, &mut room).expect("rds_demod");
            soft.extend_from_slice(&room[..p.produced]);
            rest = &rest[p.consumed..];
        }
        self.block.reset();
        self.at = 0;
        let ok = soft.len() > 500
            && soft
                .iter()
                .zip(&self.bits)
                .skip(100)
                .all(|(s, &b)| (*s > 0.0) == (b == 1));
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

/// The block `rds_decode` over `n` soft bits per call.
pub(crate) struct RdsDecodeBlockK {
    block: RdsDecodeBlock<ArenaStorage<'static>>,
    /// Two periods of the soft bits, so a call never wraps inside its input.
    soft: Vec<f32>,
    period: usize,
    n: usize,
    at: usize,
    got: Vec<Packet>,
    produced: usize,
}

impl RdsDecodeBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let bits = group_bits(rng, PERIOD_GROUPS);
        let period = bits.len();
        let mut soft: Vec<f32> = bits
            .iter()
            .map(|&b| if b == 1 { 1.0 } else { -1.0 })
            .collect();
        while soft.len() < period + layout.n {
            soft.extend_from_within(..period);
        }
        let block = RdsDecodeBlock::new(
            &RdsDecodeConfig::default(),
            Rate::hz(1187).expect("rate"),
            &mut ArenaStorage::new(&mut []),
        )
        .expect("rds_decode");
        RdsDecodeBlockK {
            block,
            soft,
            period,
            n: layout.n,
            at: 0,
            // Room for every group of a call (at most one per 104 samples).
            got: vec![Packet::default(); layout.n / 104 + 2],
            produced: 0,
        }
    }
}

#[inline(never)]
fn decode_soft(
    block: &mut RdsDecodeBlock<ArenaStorage<'static>>,
    soft: &[f32],
    got: &mut [Packet],
) -> usize {
    match block.process(black_box(soft), got) {
        Ok(p) => p.produced,
        Err(_) => 0,
    }
}

impl Kernel for RdsDecodeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    let input = &self.soft[self.at..self.at + self.n];
                    self.produced = decode_soft(&mut self.block, input, &mut self.got);
                    self.at = (self.at + self.n) % self.period;
                }
                other => unreachable!("rds_decode_block has no {other:?}"),
            }
            black_box(&self.got);
        }
    }

    /// 999 when one call from a reset gives every group of its input with `CRC_OK`.
    fn accuracy(&mut self, which: Impl) -> f64 {
        if !self.supports(which) {
            return 0.0;
        }
        self.block.reset();
        self.at = 0;
        self.run(which, 1);
        let got = &self.got[..self.produced];
        let ok = got.len() == self.n / 104
            && got
                .iter()
                .enumerate()
                .all(|(g, p)| p.flags.contains(PacketFlags::CRC_OK) && p.index == 104 * g as u64);
        self.block.reset();
        self.at = 0;
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

#[cfg(test)]
mod tests {
    use super::super::{Impl, Kernel, Layout, Pcg32};
    use super::{RdsDecodeBlockK, RdsDemodBlockK};

    /// Both entries decode their workloads (seeds 575 and 574 of the M41 range).
    #[test]
    fn the_rds_entries_decode_their_workloads() {
        let layout = Layout {
            n: 4096,
            misaligned: false,
        };
        let mut demod = RdsDemodBlockK::new(&mut Pcg32::new(575), layout);
        assert_eq!(demod.accuracy(Impl::Ours), 999.0);
        demod.run(Impl::Ours, 40);
        let mut decode = RdsDecodeBlockK::new(&mut Pcg32::new(574), layout);
        assert_eq!(decode.accuracy(Impl::Fallback), 999.0);
        decode.run(Impl::Ours, 3);
        assert!((39..=40).contains(&decode.produced));
    }
}
