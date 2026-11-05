//! The protocol entries (milestone M37): the CRCs of `sdrk-proto` and the bit-by-bit frame
//! decoder.
//!
//! | entry | measured | gate | unit of a "sample" |
//! | --- | --- | --- | --- |
//! | `crc16_ccitt_false` | `sdrk_proto::crc` | GNU Radio `gr::digital::crc` | one input byte |
//! | `crc16_x25` | `sdrk_proto::crc` | GNU Radio `gr::digital::crc` | one input byte |
//! | `crc32` | `sdrk_proto::crc` | GNU Radio `gr::digital::crc` | one input byte |
//! | `crc24_adsb` | `sdrk_proto::crc` | GNU Radio `gr::digital::crc` | one input byte |
//! | `frame_decode` | `sdrk_proto::frame::Decoder` | none (`info`) | one input bit |
//! | `frame_encode_block` | `sdrk_blocks::FrameEncodeBlock` | none (`info`) | one output sample (one bit) |
//! | `frame_decode_block` | `sdrk_blocks::FrameDecodeBlock` | none (`info`) | one input sample (one soft bit) |
//!
//! `sdrk-proto` has no backends and no SIMD (the CRCs are byte-serial table lookups over at
//! most 256 bytes, the decoder a bit-serial state machine), so `ours` and `fallback` run the
//! same code. The CRCs are gated against GNU Radio's generic table-driven CRC of gr-digital
//! (`cref_sys::GrCrc`, D527) with the same parameters and messages; the decoder and the two
//! blocks have no reference (no C library has this framing) and are information only. The
//! figures are cycles (and instructions) per byte for the CRCs and per bit for the decoder.
//!
//! Workloads: the CRCs run over `n` random bytes as messages of 256 bytes (the largest packet,
//! `MAX_PACKET_BYTES`), one call per message. The decoder gets `n` bits: frames with a 4-byte
//! preamble and a random 32-byte payload back to back (344 bits each), the rest of the `n` bits
//! zero (idle), so it spends time both hunting and inside frames; the decoder keeps its state
//! across the repeated calls as a block does.
//!
//! The two block entries (M37 blocks `frame_encode` and `frame_decode`) measure the cost per
//! signal sample of the blocks around the framing: the encoder writes `n` output samples
//! (`±1.0`, one per bit) from frames with a 4-byte preamble and a 32-byte payload (344 samples
//! each, the last frame cut at `n` and continued by the next call, as a pipeline does); the
//! decoder block reads the `n` bits of the `frame_decode` workload as `±1.0` soft bits and keeps
//! its state across calls.

use std::hint::black_box;

use sdrk::Rate;
use sdrk_blocks::{
    ArenaStorage, Block, FrameDecodeBlock, FrameDecodeConfig, FrameEncodeBlock, FrameEncodeConfig,
};
use sdrk_proto::crc;
use sdrk_proto::frame::{self, Decoder, Whitening};
use sdrk_proto::{MAX_PACKET_BYTES, Packet, PacketFlags};

use cref_sys::GrCrc;

use super::{Impl, Kernel, Layout, Pcg32};

/// One of the four CRCs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Crc {
    /// CRC-16/CCITT-FALSE.
    Ccitt,
    /// CRC-16/X-25.
    X25,
    /// CRC-32/ISO-HDLC.
    Crc32,
    /// CRC-24 of Mode S.
    Adsb,
}

impl Crc {
    fn run(self, d: &[u8]) -> u32 {
        match self {
            Crc::Ccitt => u32::from(crc::crc16_ccitt_false(d)),
            Crc::X25 => u32::from(crc::crc16_x25(d)),
            Crc::Crc32 => crc::crc32(d),
            Crc::Adsb => crc::crc24_adsb(d),
        }
    }

    /// Bit-by-bit model of the catalogue parameters (the accuracy check).
    fn model(self, d: &[u8]) -> u32 {
        // (width, poly, init, reflected, xorout)
        let (width, poly, init, reflected, xorout): (u32, u32, u32, bool, u32) = match self {
            Crc::Ccitt => (16, 0x1021, 0xFFFF, false, 0),
            Crc::X25 => (16, 0x1021, 0xFFFF, true, 0xFFFF),
            Crc::Crc32 => (32, 0x04C1_1DB7, 0xFFFF_FFFF, true, 0xFFFF_FFFF),
            Crc::Adsb => (24, 0x00FF_F409, 0, false, 0),
        };
        let mask = if width == 32 {
            u32::MAX
        } else {
            (1u32 << width) - 1
        };
        let top = 1u32 << (width - 1);
        let mut c = init;
        for &b in d {
            let b = if reflected { b.reverse_bits() } else { b };
            for k in (0..8).rev() {
                let bit = (u32::from(b) >> k) & 1;
                let fb = ((c & top) != 0) as u32 ^ bit;
                c = (c << 1) & mask;
                if fb != 0 {
                    c ^= poly;
                }
            }
        }
        if reflected {
            c = c.reverse_bits() >> (32 - width);
        }
        (c ^ xorout) & mask
    }

    /// GNU Radio's `gr::digital::crc` with the catalogue parameters; `None` without the
    /// library.
    fn gnuradio(self) -> Option<GrCrc> {
        let (width, poly, init, reflected, xorout): (u32, u64, u64, bool, u64) = match self {
            Crc::Ccitt => (16, 0x1021, 0xFFFF, false, 0),
            Crc::X25 => (16, 0x1021, 0xFFFF, true, 0xFFFF),
            Crc::Crc32 => (32, 0x04C1_1DB7, 0xFFFF_FFFF, true, 0xFFFF_FFFF),
            Crc::Adsb => (24, 0x00FF_F409, 0, false, 0),
        };
        GrCrc::new(width, poly, init, xorout, reflected, reflected)
    }
}

/// A CRC over `n` bytes in messages of 256 bytes.
pub(crate) struct CrcK {
    which: Crc,
    input: Vec<u8>,
    out: Vec<u32>,
}

impl CrcK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout, which: Crc) -> Self {
        let input: Vec<u8> = (0..layout.n).map(|_| rng.next_u32() as u8).collect();
        let out = vec![0; layout.n.div_ceil(MAX_PACKET_BYTES)];
        CrcK { which, input, out }
    }
}

#[inline(never)]
fn gr_crc_messages(gr: &mut GrCrc, input: &[u8], out: &mut [u32]) {
    for (o, m) in out
        .iter_mut()
        .zip(black_box(input).chunks(MAX_PACKET_BYTES))
    {
        // The widths are at most 32 bits, so the 64-bit remainder fits.
        *o = gr.compute(m) as u32;
    }
}

#[inline(never)]
fn crc_messages(which: Crc, input: &[u8], out: &mut [u32]) {
    for (o, m) in out
        .iter_mut()
        .zip(black_box(input).chunks(MAX_PACKET_BYTES))
    {
        *o = which.run(m);
    }
}

impl Kernel for CrcK {
    fn run(&mut self, which: Impl, k: usize) {
        let mut gr = (which == Impl::GnuRadio)
            .then(|| self.which.gnuradio())
            .flatten();
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => crc_messages(self.which, &self.input, &mut self.out),
                Impl::GnuRadio => {
                    gr_crc_messages(gr.as_mut().expect("gnuradio"), &self.input, &mut self.out)
                }
                other => unreachable!("the CRCs have no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.run(which, 1);
        let ok = self
            .input
            .chunks(MAX_PACKET_BYTES)
            .zip(&self.out)
            .all(|(m, &o)| self.which.model(m) == o);
        if ok { 999.0 } else { 0.0 }
    }

    fn supports(&self, which: Impl) -> bool {
        match which {
            Impl::Ours | Impl::Fallback => true,
            Impl::GnuRadio => self.which.gnuradio().is_some(),
            _ => false,
        }
    }

    fn describe(&self, which: Impl) -> &'static str {
        match which {
            Impl::Fallback => "sdrk fallback",
            Impl::GnuRadio => {
                "GNU Radio gr::digital::crc: one 256-entry uint64_t table, a 64-bit remainder, one lookup per byte (generic compiler flags)"
            }
            _ => "sdrk",
        }
    }
}

/// Payload bytes of the frames of the decoder workload.
const DECODE_PAYLOAD: usize = 32;
/// Preamble bytes of the frames of the decoder workload.
const DECODE_PREAMBLE: usize = 4;

/// The frame decoder over `n` bits.
pub(crate) struct FrameDecodeK {
    bits: Vec<u8>,
    payloads: Vec<Vec<u8>>,
    decoder: Decoder,
    index: u64,
    got: Vec<Packet>,
}

impl FrameDecodeK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let per = frame::frame_bits(DECODE_PAYLOAD, DECODE_PREAMBLE);
        let mut bits = Vec::with_capacity(layout.n);
        let mut payloads = Vec::new();
        while bits.len() + per <= layout.n {
            let payload: Vec<u8> = (0..DECODE_PAYLOAD).map(|_| rng.next_u32() as u8).collect();
            let p = Packet::new(0, PacketFlags::NONE, &payload).expect("packet");
            let mut f = vec![0u8; per];
            frame::encode(&p, DECODE_PREAMBLE, Whitening::None, &mut f).expect("frame");
            bits.extend(f);
            payloads.push(payload);
        }
        bits.resize(layout.n, 0);
        FrameDecodeK {
            bits,
            payloads,
            decoder: Decoder::new(0, false, Whitening::None).expect("decoder"),
            index: 0,
            got: Vec::with_capacity(64),
        }
    }
}

#[inline(never)]
fn decode_bits(dec: &mut Decoder, bits: &[u8], index: &mut u64, got: &mut Vec<Packet>) {
    for &b in black_box(bits) {
        if let Some(p) = dec.push_bit(b != 0, *index) {
            got.push(p);
        }
        *index += 1;
    }
}

impl Kernel for FrameDecodeK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            self.got.clear();
            match which {
                Impl::Ours | Impl::Fallback => {
                    decode_bits(
                        &mut self.decoder,
                        &self.bits,
                        &mut self.index,
                        &mut self.got,
                    );
                }
                other => unreachable!("frame_decode has no {other:?}"),
            }
            black_box(&self.got);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.decoder.reset();
        self.run(which, 1);
        let ok = self.got.len() == self.payloads.len()
            && self
                .got
                .iter()
                .zip(&self.payloads)
                .all(|(g, p)| g.data() == p.as_slice() && g.flags == PacketFlags::CRC_OK);
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

/// The frames of the decoder workloads: `n` bits of back-to-back frames (4-byte preamble,
/// 32-byte random payload), the rest zero, and the payloads.
fn decode_workload(rng: &mut Pcg32, n: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let per = frame::frame_bits(DECODE_PAYLOAD, DECODE_PREAMBLE);
    let mut bits = Vec::with_capacity(n);
    let mut payloads = Vec::new();
    while bits.len() + per <= n {
        let payload: Vec<u8> = (0..DECODE_PAYLOAD).map(|_| rng.next_u32() as u8).collect();
        let p = Packet::new(0, PacketFlags::NONE, &payload).expect("packet");
        let mut f = vec![0u8; per];
        frame::encode(&p, DECODE_PREAMBLE, Whitening::None, &mut f).expect("frame");
        bits.extend(f);
        payloads.push(payload);
    }
    bits.resize(n, 0);
    (bits, payloads)
}

/// The storage type of the block entries (both blocks need none).
type NoStorage = ArenaStorage<'static>;

fn rate() -> Rate {
    Rate::hz(48_000).expect("rate")
}

/// The block `frame_encode` over `n` output samples.
pub(crate) struct FrameEncodeBlockK {
    block: FrameEncodeBlock<NoStorage>,
    packets: Vec<Packet>,
    /// Packets consumed so far (the next call starts there; wraps around).
    next: usize,
    out: Vec<f32>,
    /// The expected samples of the first call (from `frame::encode`).
    want: Vec<f32>,
}

impl FrameEncodeBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let per = frame::frame_bits(DECODE_PAYLOAD, DECODE_PREAMBLE);
        // Enough packets for two calls, so the second call resumes a cut frame.
        let count = (2 * layout.n).div_ceil(per) + 1;
        let packets: Vec<Packet> = (0..count)
            .map(|_| {
                let payload: Vec<u8> = (0..DECODE_PAYLOAD).map(|_| rng.next_u32() as u8).collect();
                Packet::new(0, PacketFlags::NONE, &payload).expect("packet")
            })
            .collect();
        let mut want = Vec::with_capacity(count * per);
        for p in &packets {
            let mut f = vec![0u8; per];
            frame::encode(p, DECODE_PREAMBLE, Whitening::None, &mut f).expect("frame");
            want.extend(f.iter().map(|&b| if b == 1 { 1.0f32 } else { -1.0 }));
        }
        want.truncate(layout.n);
        let cfg = FrameEncodeConfig {
            preamble_bytes: DECODE_PREAMBLE,
            ..FrameEncodeConfig::default()
        };
        let block = FrameEncodeBlock::new(&cfg, rate(), &mut ArenaStorage::new(&mut []))
            .expect("frame_encode");
        FrameEncodeBlockK {
            block,
            packets,
            next: 0,
            out: vec![0.0; layout.n],
            want,
        }
    }
}

#[inline(never)]
fn encode_samples(
    block: &mut FrameEncodeBlock<NoStorage>,
    packets: &[Packet],
    out: &mut [f32],
) -> usize {
    match block.process(black_box(packets), out) {
        Ok(p) => p.consumed,
        Err(_) => usize::MAX,
    }
}

impl Kernel for FrameEncodeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    if self.next >= self.packets.len() / 2 {
                        self.block.reset();
                        self.next = 0;
                    }
                    let consumed =
                        encode_samples(&mut self.block, &self.packets[self.next..], &mut self.out);
                    self.next += consumed;
                }
                other => unreachable!("frame_encode_block has no {other:?}"),
            }
            black_box(&self.out);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.block.reset();
        self.next = 0;
        self.run(which, 1);
        if self.out == self.want { 999.0 } else { 0.0 }
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

/// The block `frame_decode` over `n` soft bits.
pub(crate) struct FrameDecodeBlockK {
    block: FrameDecodeBlock<NoStorage>,
    soft: Vec<f32>,
    payloads: Vec<Vec<u8>>,
    got: Vec<Packet>,
    produced: usize,
}

impl FrameDecodeBlockK {
    pub(crate) fn new(rng: &mut Pcg32, layout: Layout) -> Self {
        let (bits, payloads) = decode_workload(rng, layout.n);
        let soft = bits
            .iter()
            .map(|&b| if b == 1 { 1.0f32 } else { -1.0 })
            .collect();
        let block = FrameDecodeBlock::new(
            &FrameDecodeConfig::default(),
            rate(),
            &mut ArenaStorage::new(&mut []),
        )
        .expect("frame_decode");
        FrameDecodeBlockK {
            block,
            soft,
            payloads,
            // Room for every frame of a call (at most one per 56 samples).
            got: vec![Packet::default(); layout.n / 56 + 1],
            produced: 0,
        }
    }
}

#[inline(never)]
fn decode_soft(block: &mut FrameDecodeBlock<NoStorage>, soft: &[f32], got: &mut [Packet]) -> usize {
    match block.process(black_box(soft), got) {
        Ok(p) => p.produced,
        Err(_) => 0,
    }
}

impl Kernel for FrameDecodeBlockK {
    fn run(&mut self, which: Impl, k: usize) {
        for _ in 0..k {
            match which {
                Impl::Ours | Impl::Fallback => {
                    self.produced = decode_soft(&mut self.block, &self.soft, &mut self.got);
                }
                other => unreachable!("frame_decode_block has no {other:?}"),
            }
            black_box(&self.got);
        }
    }

    fn accuracy(&mut self, which: Impl) -> f64 {
        self.block.reset();
        self.run(which, 1);
        let got = &self.got[..self.produced];
        let ok = got.len() == self.payloads.len()
            && got
                .iter()
                .zip(&self.payloads)
                .all(|(g, p)| g.data() == p.as_slice() && g.flags == PacketFlags::CRC_OK);
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
    use super::super::{Kernel, Layout, Pcg32};
    use super::{Crc, CrcK, Impl};

    /// Every CRC entry agrees with the bit-by-bit model for each implementation it supports,
    /// GNU Radio's `gr::digital::crc` included when the reference is linked.
    #[test]
    fn the_crc_entries_match_the_model() {
        for which in [Crc::Ccitt, Crc::X25, Crc::Crc32, Crc::Adsb] {
            let mut rng = Pcg32::new(479);
            let layout = Layout {
                n: 1000,
                misaligned: false,
            };
            let mut kernel = CrcK::new(&mut rng, layout, which);
            for imp in [Impl::Ours, Impl::Fallback, Impl::GnuRadio] {
                if kernel.supports(imp) {
                    assert_eq!(kernel.accuracy(imp), 999.0, "{which:?} {}", imp.name());
                }
            }
            assert_eq!(
                kernel.supports(Impl::GnuRadio),
                cref_sys::GrCrc::new(16, 0x1021, 0, 0, false, false).is_some()
            );
        }
    }
}
